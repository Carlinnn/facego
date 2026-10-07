package main

import (
	"fmt"
	"log"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: inspect_onnx <model.onnx>")
	}
	modelPath := os.Args[1]

	ort.SetSharedLibraryPath("onnxruntime.dll")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("init ort: %v", err)
	}
	defer ort.DestroyEnvironment()

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		log.Fatalf("get model info: %v", err)
	}

	fmt.Printf("Model: %s\n\n", modelPath)
	fmt.Println("=== INPUTS ===")
	for _, in := range inputs {
		fmt.Printf("  name=%q  shape=%v  type=%v\n", in.Name, in.Dimensions, in.DataType)
	}
	fmt.Println("\n=== OUTPUTS ===")
	for _, out := range outputs {
		fmt.Printf("  name=%q  shape=%v  type=%v\n", out.Name, out.Dimensions, out.DataType)
	}
}
