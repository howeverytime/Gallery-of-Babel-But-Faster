// Вавилонская галерея: все возможные картинки из сетки цветных клеток.
//
// Экспонат = битовая строка: пиксель №p занимает биты [p*bpp, (p+1)*bpp).
// ID экспоната = эта строка в base64url. Никакой длинной арифметики:
// «следующий экспонат» — это +1 с переносом по байтам, «случайный» — просто случайные байты.
//
// Сервер + веб-интерфейс (всё рисуется в браузере, сервер почти ничего не делает):
//
//	go run main.go                      (или: go run main.go -serve -port 8080)
//
// CLI (картинка в файл):
//
//	go run main.go -o out.png -w 64 -h 64 -bpp 8 -scale 8
//	go run main.go -o out.png -seed 42          (воспроизводимый экспонат)
//	go run main.go -o out.png -id <ID> -w 64 -h 64 -bpp 8
//
// Видео → txt с ID кадров (нужен ffmpeg; берём каждый N-й кадр, пиксели усредняются):
//
//	go run main.go -video clip.mp4 -w 96 -every 5 -bpp 8 -o clip.txt
//	(-h не указан → высота считается по пропорциям видео; -ss 10 -t 30 — взять отрезок)
//
// Полученный txt вставляется в веб-интерфейс («Найти экспонат» → вставить текст или
// выбрать файл) и проигрывается как фильм.
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/base64"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io"
	"log"
	"math"
	mrand "math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------- модель экспоната ----------

type spec struct{ w, h, bpp int }

const maxSide = 512

func (s spec) check() error {
	if s.w < 1 || s.h < 1 || s.w > maxSide || s.h > maxSide {
		return fmt.Errorf("ширина и высота должны быть от 1 до %d", maxSide)
	}
	switch s.bpp {
	case 1, 2, 4, 8, 12, 24:
		return nil
	}
	return errors.New("bpp должен быть одним из: 1, 2, 4, 8, 12, 24")
}

func (s spec) bits() int   { return s.w * s.h * s.bpp }
func (s spec) nbytes() int { return (s.bits() + 7) / 8 }

// maskPad обнуляет неиспользуемые младшие биты последнего байта.
func (s spec) maskPad(b []byte) {
	pad := uint(s.nbytes()*8 - s.bits())
	b[len(b)-1] &= byte(0xFF) << pad
}

func (s spec) random(seed *int64) []byte {
	b := make([]byte, s.nbytes())
	if seed != nil {
		mrand.New(mrand.NewSource(*seed)).Read(b)
	} else if _, err := rand.Read(b); err != nil {
		log.Fatal(err)
	}
	s.maskPad(b)
	return b
}

func (s spec) decodeID(id string) ([]byte, error) {
	b, err := base64.RawURLEncoding.DecodeString(strings.TrimSpace(strings.TrimPrefix(id, "#")))
	if err != nil {
		return nil, errors.New("ID не похож на base64url")
	}
	if len(b) != s.nbytes() {
		return nil, fmt.Errorf("для %dx%d, %d бит/пикс нужен ID из %d байт, а тут %d",
			s.w, s.h, s.bpp, s.nbytes(), len(b))
	}
	s.maskPad(b)
	return b, nil
}

// ---------- палитра (должна совпадать с JS) ----------

func ex(c, max int) uint8 { return uint8((c*255 + max/2) / max) }

func rgbOf(bpp int, v uint32) (r, g, b uint8) {
	switch bpp {
	case 1:
		if v != 0 {
			return 255, 255, 255
		}
		return 0, 0, 0
	case 2:
		x := uint8(v * 85)
		return x, x, x
	case 4: // 16 цветов EGA: биты R,G,B + бит яркости
		i := uint8((v>>3)&1) * 85
		return uint8((v>>2)&1)*170 + i, uint8((v>>1)&1)*170 + i, uint8(v&1)*170 + i
	case 8: // 3-3-2
		return ex(int(v>>5), 7), ex(int((v>>2)&7), 7), ex(int(v&3), 3)
	case 12: // 4-4-4
		return uint8(v>>8) * 17, uint8((v>>4)&15) * 17, uint8(v&15) * 17
	default: // 24
		return uint8(v >> 16), uint8(v >> 8), uint8(v)
	}
}

func buildLUT(bpp int) [][3]uint8 {
	if bpp > 12 {
		return nil
	}
	lut := make([][3]uint8, 1<<uint(bpp))
	for v := range lut {
		r, g, b := rgbOf(bpp, uint32(v))
		lut[v] = [3]uint8{r, g, b}
	}
	return lut
}

func pixelAt(b []byte, p, bpp int) uint32 {
	switch bpp {
	case 8:
		return uint32(b[p])
	case 24:
		o := p * 3
		return uint32(b[o])<<16 | uint32(b[o+1])<<8 | uint32(b[o+2])
	}
	o := p * bpp
	var v uint32
	for need := bpp; need > 0; {
		used := o & 7
		take := 8 - used
		if take > need {
			take = need
		}
		v = v<<uint(take) | (uint32(b[o>>3])>>uint(8-used-take))&((uint32(1)<<uint(take))-1)
		o += take
		need -= take
	}
	return v
}

// ---------- рендер ----------

// renderImage рисует экспонат, каждая клетка — scale×scale пикселей.
// Строки сетки делятся между горутинами, пишем прямо в Pix.
func renderImage(s spec, b []byte, scale int) *image.RGBA {
	W, H := s.w*scale, s.h*scale
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	lut := buildLUT(s.bpp)

	workers := runtime.NumCPU()
	if workers > s.h {
		workers = s.h
	}
	chunk := (s.h + workers - 1) / workers
	var wg sync.WaitGroup
	for y0 := 0; y0 < s.h; y0 += chunk {
		y1 := y0 + chunk
		if y1 > s.h {
			y1 = s.h
		}
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			for y := y0; y < y1; y++ {
				base := y * scale * img.Stride
				row := img.Pix[base : base+W*4]
				for x := 0; x < s.w; x++ {
					v := pixelAt(b, y*s.w+x, s.bpp)
					var r, g, bl uint8
					if lut != nil {
						c := lut[v]
						r, g, bl = c[0], c[1], c[2]
					} else {
						r, g, bl = rgbOf(s.bpp, v)
					}
					for k := 0; k < scale; k++ {
						o := (x*scale + k) * 4
						row[o], row[o+1], row[o+2], row[o+3] = r, g, bl, 255
					}
				}
				for k := 1; k < scale; k++ {
					copy(img.Pix[base+k*img.Stride:], row)
				}
			}
		}(y0, y1)
	}
	wg.Wait()
	return img
}

var fastPNG = png.Encoder{CompressionLevel: png.BestSpeed}

// ---------- main ----------

func main() {
	w := flag.Int("w", 64, "ширина сетки (клеток)")
	h := flag.Int("h", 64, "высота сетки (клеток)")
	bpp := flag.Int("bpp", 8, "бит на клетку: 1, 2, 4, 8, 12 или 24")
	id := flag.String("id", "", "ID экспоната (base64url); по умолчанию случайный")
	seed := flag.Int64("seed", 0, "seed для воспроизводимого экспоната")
	scale := flag.Int("scale", 0, "размер клетки в пикселях (0 = авто, ~1024 px)")
	out := flag.String("o", "", "куда сохранить PNG (включает режим CLI)")
	printID := flag.Bool("printid", false, "напечатать полный ID")
	video := flag.String("video", "", "видеофайл → txt с ID кадров (нужен ffmpeg); -o = куда писать txt")
	every := flag.Int("every", 5, "видео: брать каждый N-й кадр")
	ss := flag.String("ss", "", "видео: начать с этого момента (сек или чч:мм:сс)")
	dur := flag.String("t", "", "видео: длительность (сек или чч:мм:сс)")
	serve := flag.Bool("serve", false, "запустить веб-сервер")
	port := flag.Int("port", 8080, "порт сервера")
	flag.Parse()

	set := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { set[f.Name] = true })
	cli := set["o"] || set["id"] || set["seed"] || set["printid"] || set["video"]
	if *serve || !cli {
		runServer(*port)
		return
	}

	s := spec{*w, *h, *bpp}
	if *video != "" {
		outPath := *out
		if outPath == "" {
			outPath = strings.TrimSuffix(*video, filepath.Ext(*video)) + "_babel.txt"
		}
		if err := runVideoCLI(*video, s, !set["h"], *every, *ss, *dur, outPath); err != nil {
			log.Fatal(err)
		}
		return
	}
	if err := s.check(); err != nil {
		log.Fatal(err)
	}
	var b []byte
	switch {
	case *id != "":
		var err error
		if b, err = s.decodeID(*id); err != nil {
			log.Fatal(err)
		}
	case set["seed"]:
		b = s.random(seed)
	default:
		b = s.random(nil)
	}
	sc := *scale
	if sc <= 0 {
		sc = autoScale(s)
	}
	if *out == "" {
		*out = "exhibit.png"
	}
	f, err := os.Create(*out)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	if err := fastPNG.Encode(f, renderImage(s, b, sc)); err != nil {
		log.Fatal(err)
	}
	full := base64.RawURLEncoding.EncodeToString(b)
	if *printID || len(full) <= 40 {
		fmt.Println("ID:", full)
	} else {
		fmt.Printf("ID: %s...%s (%d симв., -printid чтобы вывести целиком)\n", full[:8], full[len(full)-8:], len(full))
	}
	fmt.Printf("экспонатов в галерее: ~10^%d\n", int(float64(s.bits())*math.Log10(2)))
	fmt.Println("готово:", *out)
}

func autoScale(s spec) int {
	m := s.w
	if s.h > m {
		m = s.h
	}
	sc := (1024 + m - 1) / m
	if sc < 1 {
		sc = 1
	}
	return sc
}

// ---------- видео ----------

type videoInfo struct {
	w, h int
	fps  float64
}

func needFFmpeg() error {
	for _, n := range []string{"ffmpeg", "ffprobe"} {
		if _, err := exec.LookPath(n); err != nil {
			return fmt.Errorf("не найден %s: установите ffmpeg (apt install ffmpeg / pkg install ffmpeg / brew install ffmpeg)", n)
		}
	}
	return nil
}

func parseRate(v string) float64 {
	a, b, ok := strings.Cut(v, "/")
	if !ok {
		f, _ := strconv.ParseFloat(v, 64)
		return f
	}
	x, _ := strconv.ParseFloat(a, 64)
	y, _ := strconv.ParseFloat(b, 64)
	if y == 0 {
		return 0
	}
	return x / y
}

// probeVideo узнаёт размеры (с учётом поворота телефонного видео) и fps.
func probeVideo(path string) (videoInfo, error) {
	if err := needFFmpeg(); err != nil {
		return videoInfo{}, err
	}
	out, err := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0",
		"-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:stream_tags=rotate:stream_side_data=rotation",
		"-of", "default=noprint_wrappers=1", path).Output()
	if err != nil {
		return videoInfo{}, errors.New("не удалось прочитать видео (ffprobe)")
	}
	kv := map[string]string{}
	for _, ln := range strings.Split(string(out), "\n") {
		if k, v, ok := strings.Cut(strings.TrimSpace(ln), "="); ok {
			kv[k] = v
		}
	}
	info := videoInfo{}
	info.w, _ = strconv.Atoi(kv["width"])
	info.h, _ = strconv.Atoi(kv["height"])
	if info.w < 1 || info.h < 1 {
		return info, errors.New("в файле нет видеодорожки")
	}
	rot := kv["rotation"]
	if rot == "" {
		rot = kv["TAG:rotate"]
	}
	if r, err := strconv.ParseFloat(rot, 64); err == nil && int(math.Round(math.Abs(r)/90))%2 == 1 {
		info.w, info.h = info.h, info.w
	}
	info.fps = parseRate(kv["avg_frame_rate"])
	if info.fps <= 0 {
		info.fps = parseRate(kv["r_frame_rate"])
	}
	if info.fps <= 0 || info.fps > 1000 {
		info.fps = 25
	}
	return info, nil
}

func autoHeight(w int, info videoInfo) int {
	h := int(math.Round(float64(w) * float64(info.h) / float64(info.w)))
	if h < 1 {
		h = 1
	}
	if h > maxSide {
		h = maxSide
	}
	return h
}

// ---- квантование RGB → индекс палитры (для видео) ----

var (
	egaOnce sync.Once
	egaLUT  [32768]uint8
)

func egaIndex(r, g, b int) uint32 {
	egaOnce.Do(func() {
		for i := range egaLUT {
			cr, cg, cb := (i>>10)<<3|4, ((i>>5)&31)<<3|4, (i&31)<<3|4
			best, bd := 0, 1<<30
			for v := 0; v < 16; v++ {
				pr, pg, pb := rgbOf(4, uint32(v))
				dr, dg, db := cr-int(pr), cg-int(pg), cb-int(pb)
				if d := dr*dr + dg*dg + db*db; d < bd {
					bd, best = d, v
				}
			}
			egaLUT[i] = uint8(best)
		}
	})
	return uint32(egaLUT[(r>>3)<<10|(g>>3)<<5|(b>>3)])
}

func quantRGB(bpp, r, g, b int) uint32 {
	switch bpp {
	case 1:
		if 0.299*float64(r)+0.587*float64(g)+0.114*float64(b) >= 128 {
			return 1
		}
		return 0
	case 2:
		return uint32((0.299*float64(r)+0.587*float64(g)+0.114*float64(b))/85 + 0.5)
	case 4:
		return egaIndex(r, g, b)
	case 8:
		return uint32(((r*14+255)/510)<<5 | ((g*14+255)/510)<<2 | (b*6+255)/510)
	case 12:
		return uint32(((r*2+17)/34)<<8 | ((g*2+17)/34)<<4 | (b*2+17)/34)
	}
	return uint32(r<<16 | g<<8 | b)
}

func setBits(b []byte, p, bpp int, v uint32) {
	for i := 0; i < bpp; i++ {
		pos := p*bpp + i
		m := byte(0x80) >> uint(pos&7)
		if (v>>uint(bpp-1-i))&1 != 0 {
			b[pos>>3] |= m
		} else {
			b[pos>>3] &^= m
		}
	}
}

// packFrame: кадр RGB24 (w*h*3) → битовая строка экспоната.
func packFrame(rgb []byte, s spec, out []byte) {
	n := s.w * s.h
	switch s.bpp {
	case 24:
		copy(out, rgb)
		return
	case 8:
		for p := 0; p < n; p++ {
			o := p * 3
			out[p] = byte(quantRGB(8, int(rgb[o]), int(rgb[o+1]), int(rgb[o+2])))
		}
		return
	}
	for i := range out {
		out[i] = 0
	}
	for p := 0; p < n; p++ {
		o := p * 3
		setBits(out, p, s.bpp, quantRGB(s.bpp, int(rgb[o]), int(rgb[o+1]), int(rgb[o+2])))
	}
}

// passthroughFlags: без этого ffmpeg «добивает» пропущенные кадры дубликатами, чтобы держать
// постоянный fps (вместо 24 кадров из 120 при every=5 получилось бы 119). Название опции
// зависит от версии ffmpeg: новое -fps_mode, старое -vsync 0.
var (
	ptOnce  sync.Once
	ptFlags []string
)

func passthroughFlags() []string {
	ptOnce.Do(func() {
		ptFlags = []string{"-vsync", "0"}
		err := exec.Command("ffmpeg", "-v", "quiet", "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.1",
			"-fps_mode", "passthrough", "-f", "null", "-").Run()
		if err == nil {
			ptFlags = []string{"-fps_mode", "passthrough"}
		}
	})
	return ptFlags
}

// convertVideo: ffmpeg отбирает каждый every-й кадр ДО масштабирования (остальные не
// обрабатываются вообще), scale=area усредняет пиксели, rawvideo идёт по пайпу; здесь
// кадр квантуется и пишется одной строкой base64url.
func convertVideo(ctx context.Context, path string, s spec, every int, ss, dur string,
	info videoInfo, out io.Writer, progress func(n int)) (int, error) {
	if every < 1 {
		every = 1
	}
	args := []string{"-v", "error"}
	if ss != "" {
		args = append(args, "-ss", ss)
	}
	args = append(args, "-i", path)
	if dur != "" {
		args = append(args, "-t", dur)
	}
	args = append(args, passthroughFlags()...)
	args = append(args,
		"-vf", fmt.Sprintf("select=not(mod(n\\,%d)),scale=%d:%d:flags=area", every, s.w, s.h),
		"-pix_fmt", "rgb24", "-f", "rawvideo", "-")

	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return 0, err
	}
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	rd := bufio.NewReaderSize(stdout, 1<<20)
	bw := bufio.NewWriterSize(out, 1<<20)
	fmt.Fprintf(bw, "#babel1 w=%d h=%d b=%d fps=%.3f every=%d\n", s.w, s.h, s.bpp, info.fps/float64(every), every)

	frame := make([]byte, s.w*s.h*3)
	packed := make([]byte, s.nbytes())
	line := make([]byte, base64.RawURLEncoding.EncodedLen(len(packed))+1)
	n := 0
	for {
		if _, err := io.ReadFull(rd, frame); err != nil {
			break
		}
		packFrame(frame, s, packed)
		s.maskPad(packed)
		base64.RawURLEncoding.Encode(line, packed)
		line[len(line)-1] = '\n'
		if _, err := bw.Write(line); err != nil {
			cmd.Process.Kill()
			cmd.Wait()
			return n, err
		}
		n++
		if progress != nil && n%25 == 0 {
			progress(n)
		}
	}
	werr := cmd.Wait()
	if err := bw.Flush(); err != nil {
		return n, err
	}
	if n == 0 {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" && werr != nil {
			msg = werr.Error()
		}
		if msg == "" {
			msg = "ffmpeg не выдал ни одного кадра (слишком большой -every или пустой отрезок?)"
		}
		return 0, errors.New(msg)
	}
	return n, nil
}

func runVideoCLI(path string, s spec, autoH bool, every int, ss, dur, outPath string) error {
	info, err := probeVideo(path)
	if err != nil {
		return err
	}
	if autoH {
		s.h = autoHeight(s.w, info)
	}
	if err := s.check(); err != nil {
		return err
	}
	if every < 1 {
		return errors.New("-every должен быть >= 1")
	}
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()
	start := time.Now()
	n, err := convertVideo(context.Background(), path, s, every, ss, dur, info, f,
		func(n int) { fmt.Fprintf(os.Stderr, "\rкадров: %d", n) })
	fmt.Fprint(os.Stderr, "\r")
	if err != nil {
		return err
	}
	st, _ := f.Stat()
	fmt.Printf("готово: %s\n", outPath)
	fmt.Printf("кадров: %d (каждый %d-й из ~%.1f к/с), сетка %dx%d, %d бит/клетка, воспроизведение %.2f к/с\n",
		n, every, info.fps, s.w, s.h, s.bpp, info.fps/float64(every))
	fmt.Printf("размер: %.1f МБ, время: %.1f с\n", float64(st.Size())/1e6, time.Since(start).Seconds())
	return nil
}

// ---------- сервер ----------

func runServer(port int) {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(indexHTML))
	})
	mux.HandleFunc("/img.png", handleImg)
	mux.HandleFunc("/api/random", handleRandom)
	mux.HandleFunc("/api/video", handleVideo)

	addr := ":" + strconv.Itoa(port)
	log.Printf("галерея: http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func qInt(r *http.Request, key string, def int) int {
	if v, err := strconv.Atoi(r.URL.Query().Get(key)); err == nil {
		return v
	}
	return def
}

func specFrom(r *http.Request) (spec, error) {
	s := spec{qInt(r, "w", 64), qInt(r, "h", 64), qInt(r, "b", 8)}
	return s, s.check()
}

// GET /img.png?w=64&h=64&b=8&id=...&scale=8 — PNG экспоната (без id — случайный).
func handleImg(w http.ResponseWriter, r *http.Request) {
	s, err := specFrom(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var b []byte
	if id := r.URL.Query().Get("id"); id != "" {
		if b, err = s.decodeID(id); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	} else {
		b = s.random(nil)
		w.Header().Set("Cache-Control", "no-store")
	}
	sc := qInt(r, "scale", autoScale(s))
	if sc < 1 || s.w*sc > 8192 || s.h*sc > 8192 {
		http.Error(w, "scale вне диапазона (результат до 8192 px)", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "image/png")
	if err := fastPNG.Encode(w, renderImage(s, b, sc)); err != nil {
		log.Println("png:", err)
	}
}

// GET /api/random?w=&h=&b= — случайный ID (текст).
func handleRandom(w http.ResponseWriter, r *http.Request) {
	s, err := specFrom(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.Write([]byte(base64.RawURLEncoding.EncodeToString(s.random(nil))))
}

var (
	videoSem = make(chan struct{}, 2)
	timeRe   = regexp.MustCompile(`^[0-9][0-9:.]*$`)
)

func fInt(r *http.Request, key string, def int) int {
	if v, err := strconv.Atoi(r.FormValue(key)); err == nil {
		return v
	}
	return def
}

// POST /api/video (multipart: v=файл, w, h (0=по пропорциям), b, every, ss, t) → text/plain со всеми ID.
func handleVideo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "нужен POST", http.StatusMethodNotAllowed)
		return
	}
	select {
	case videoSem <- struct{}{}:
		defer func() { <-videoSem }()
	default:
		http.Error(w, "сервер занят другими конвертациями, попробуйте чуть позже", http.StatusTooManyRequests)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 2<<30)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "не удалось принять файл (до 2 ГБ)", http.StatusBadRequest)
		return
	}
	defer r.MultipartForm.RemoveAll()
	up, _, err := r.FormFile("v")
	if err != nil {
		http.Error(w, "нет файла v", http.StatusBadRequest)
		return
	}
	defer up.Close()

	tmp, err := os.CreateTemp("", "babel-in-*")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer os.Remove(tmp.Name())
	_, err = io.Copy(tmp, up)
	tmp.Close()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ss, dur := r.FormValue("ss"), r.FormValue("t")
	if (ss != "" && !timeRe.MatchString(ss)) || (dur != "" && !timeRe.MatchString(dur)) {
		http.Error(w, "неверный формат времени", http.StatusBadRequest)
		return
	}
	every := fInt(r, "every", 5)
	if every < 1 || every > 10000 {
		http.Error(w, "every: от 1 до 10000", http.StatusBadRequest)
		return
	}
	info, err := probeVideo(tmp.Name())
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s := spec{fInt(r, "w", 64), fInt(r, "h", 0), fInt(r, "b", 8)}
	if s.h <= 0 {
		s.h = autoHeight(s.w, info)
	}
	if err := s.check(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// сначала в файл, чтобы при ошибке ffmpeg вернуть нормальный статус и Content-Length
	outF, err := os.CreateTemp("", "babel-out-*")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer os.Remove(outF.Name())
	defer outF.Close()
	if _, err := convertVideo(r.Context(), tmp.Name(), s, every, ss, dur, info, outF, nil); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="babel-frames.txt"`)
	w.Header().Set("Cache-Control", "no-store")
	http.ServeContent(w, r, "babel-frames.txt", time.Time{}, outF)
}

// ---------- веб-интерфейс (index.html, встроен в бинарник) ----------

//go:embed index.html
var indexHTML string
