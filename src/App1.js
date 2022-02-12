// Import dependencies
import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";
// import { nextFrame } from "@tensorflow/tfjs";
// 2. TODO - Import drawing utility here
// e.g. import { drawRect } from "./utilities";
import { drawRect } from "./utilities";
// import axios from "axios";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Main function
  const runCoco = async () => {
    // 3. TODO - Load network
    // e.g. const net = await cocossd.load();
    // https://tensorflowjsrealtimemodel.s3.au-syd.cloud-object-storage.appdomain.cloud/model.json
    // https://bisindo-surakarta.com/uploads/model/new/model.json

    // const net = await tf.loadGraphModel(
    //   "http://bisandro.com//uploads/model/model.json"
    //   // "https://bisindo-surakarta.com/uploads/model/model.json"
    // );

    const net = await tf.loadLayersModel(
      // "http://bisandro.com//uploads/model/model.json"
      // "https://bisindo-surakarta.com/uploads/model/model.json"
      "http://bisandro.com/uploads/model/model.json"
      // "http://bisandro.com/uploads/rescale/model.json"
      // "http://localhost:1234"
      // "/home/none/Project/bisandro-rest/mobile/ReactComputerVisionTemplate/src/layermodel/model/model.json"
    );
    // console.log(net)

    // axios.get('http://localhost:1234').then(res=>{
    //   console.log(res);
    // });

    //  Loop and detect hands
    // setInterval(() => {
    //   detect(net);
    // }, 16.7);
    detect(net);
  };

  const detect = async net => {
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // 4. TODO - Make Detections
      const img = tf.browser.fromPixels(video);
      // const resized = tf.image.resizeBilinear(img, [640, 480]);
      const resized = tf.image.resizeBilinear(img, [180, 180]);
      const casted = resized.cast("int32");
      const expanded = casted.expandDims(0);
      // const obj = await net.executeAsync(expanded);
      const obj = await net.predict(expanded);
      console.log("agdshs")
      console.log(obj);

      const boxes = await obj[1].array();
      const classes = await obj[2].array();
      const scores = await obj[4].array();

      // Draw mesh
      const ctx = canvasRef.current.getContext("2d");

      // 5. TODO - Update drawing utility
      // drawSomething(obj, ctx)
      requestAnimationFrame(() => {
        drawRect(
          boxes[0],
          classes[0],
          scores[0],
          0.8,
          videoWidth,
          videoHeight,
          ctx
        );
      });

      tf.dispose(img);
      tf.dispose(resized);
      tf.dispose(casted);
      tf.dispose(expanded);
      tf.dispose(obj);
    }
  };

  useEffect(() => {
    runCoco();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          muted={true}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 8,
            width: 640,
            height: 480
          }}
        />
      </header>
    </div>
  );
}

export default App;