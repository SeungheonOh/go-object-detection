package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"os"
	_ "strconv"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	op "github.com/tensorflow/tensorflow/tensorflow/go/op"

	cv "gocv.io/x/gocv"
)

const (
	//graphFile  = "./frozen_inference_graph.pb"
	graphFile  = "./obj/frozen_inference_graph.pb"
	labelsFile = "./coco_labels.txt"
	/*
		graphFile  = "./CSGO_frozen_inference_graph.pb"
		labelsFile = "./CSGO_labelmap.txt"
	*/
)

func Normalize(tensorIn *tf.Tensor, cols, rows int) (*tf.Tensor, error) {

	s := op.NewScope()
	i := op.Placeholder(s.SubScope("input"), tf.Uint8)
	o := op.ExpandDims(s,
		op.Reshape(s, i, op.Const(s.SubScope("make_batch"), []int32{int32(cols), int32(rows), 3})),
		op.Const(s.SubScope("make_batch"), int32(0)))

	g, err := s.Finalize()
	if err != nil {
		return nil, err
	}

	Sess, err := tf.NewSession(g, nil)
	defer Sess.Close()

	norm, err := Sess.Run(
		map[tf.Output]*tf.Tensor{i: tensorIn},
		[]tf.Output{o},
		nil)

	if err != nil {
		return nil, err
	}

	return norm[0], nil
}

func main() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")

	/*
		cam, _ := cv.VideoCaptureDevice(0)
		defer cam.Close()
	*/

	Graber := NewXScreenGraber()
	defer Graber.Close()
	_ = Graber

	window := cv.NewWindow("cv")
	defer window.Close()

	modelGraph, labels, err := loadModel()
	if err != nil {
		log.Fatalf("unable to load model: %v", err)
	}

	Session, err := tf.NewSession(modelGraph, nil)
	if err != nil {
		log.Fatalf("unable to load model: %v", err)
	}
	_ = labels
	//fmt.Println(labels)

	/*
		WinId, err := strconv.ParseUint(os.Args[1], 16, 64)
		if err != nil {
			panic(err)
		}
		fmt.Println(WinId)
	*/

	img := cv.NewMat()
	defer img.Close()

	fmt.Println("Loop start here")
	for {
		//capture := Graber.GrabById(WinId)
		capture := Graber.Grab(0, 0, 2560, 1440)

		//fmt.Println(len(imgBytes), len(arr))
		img, err := cv.NewMatFromBytes(capture.Height, capture.Width, cv.MatTypeCV8UC3, capture.ToRGB())
		if err != nil {
			panic(err)
		}

		//cam.Read(&img)
		imginput, err := tf.NewTensor(img.ToBytes())
		if err != nil {
			panic(err)
		}

		normTensor, err := Normalize(imginput, img.Rows(), img.Cols())
		if err != nil {
			panic(err)
		}

		graph := modelGraph

		// Get all the input and output operations
		inputop := graph.Operation("image_tensor")
		// Output ops
		o1 := graph.Operation("detection_boxes")
		o2 := graph.Operation("detection_scores")
		o3 := graph.Operation("detection_classes")
		o4 := graph.Operation("num_detections")

		output, err := Session.Run(
			map[tf.Output]*tf.Tensor{
				inputop.Output(0): normTensor,
			},
			[]tf.Output{
				o1.Output(0),
				o2.Output(0),
				o3.Output(0),
				o4.Output(0),
			},
			nil)
		if err != nil {
			log.Fatalf("Error running session: %v", err)
		}

		boxes := output[0].Value().([][][]float32)[0]
		classes := output[2].Value().([][]float32)[0]
		probabilities := output[1].Value().([][]float32)[0]

		for i := 0; i < len(classes); i++ {
			if probabilities[i] > 0.50 && int(classes[i]) < len(labels) {
				fmt.Println(labels[int(classes[i])])
				xMin := int(boxes[i][1] * float32(img.Cols()))
				yMin := int(boxes[i][0] * float32(img.Rows()))
				xMax := int(boxes[i][3] * float32(img.Cols()))
				yMax := int(boxes[i][2] * float32(img.Rows()))
				cv.Rectangle(&img, image.Rectangle{image.Point{xMin, yMin}, image.Point{xMax, yMax}}, color.RGBA{0, 0, 255, 255}, 4)
			}
		}
		//fmt.Println("\n", output[3].Value().([]float32)[0], "\n")

		window.IMShow(img)
		window.WaitKey(1)
	}
}

func loadModel() (*tf.Graph, []string, error) {
	// Load inception model
	model, err := ioutil.ReadFile(graphFile)
	if err != nil {
		return nil, nil, err
	}
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, nil, err
	}

	// Load labels
	labelsFile, err := os.Open(labelsFile)
	if err != nil {
		return nil, nil, err
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	return graph, labels, scanner.Err()
}
