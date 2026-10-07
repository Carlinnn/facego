package face

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

var ortInitialized bool

func InitORT(libPath string) error {
	if ortInitialized {
		return nil
	}
	if libPath == "" {
		libPath = defaultORTLib()
	}
	if _, err := os.Stat(libPath); err != nil {
		return fmt.Errorf("onnxruntime library not found at %s", libPath)
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("init ort: %w", err)
	}
	ortInitialized = true
	return nil
}

func defaultORTLib() string {
	name := "libonnxruntime.so"
	if runtime.GOOS == "windows" {
		name = "onnxruntime.dll"
	} else if runtime.GOOS == "darwin" {
		name = "libonnxruntime.dylib"
	}
	exe, _ := os.Executable()
	local := filepath.Join(filepath.Dir(exe), name)
	if _, err := os.Stat(local); err == nil {
		return local
	}
	return name
}
