// Import dependencies
import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";
// import { nextFrame } from "@tensorflow/tfjs";
// 2. TODO - Import drawing utility here
// e.g. import { drawRect } from "./utilities";
// import { drawRect } from "./utilities";
// import axios from "axios";

// import UseScript from "./UseScript";

const className = ["Aku", "Apa", "Apakabar", "Baik", "Kamu", "Nama", "Siapa"];

function App() {
  const webcamRef = useRef(null);
  // const canvasRef = useRef(null);
  const [hasil, setHasil] = useState([]);

  // UseScript(
  //   "https://ajax.aspnetcdn.com/ajax/jquery.ui/1.8.10/jquery-ui.min.js"
  // );

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
      "https://bisandro.com/uploads/model/model.json"
      // "http://bisandro.com/uploads/rescale/model.json"
      // "http://localhost:1234"
      // "/home/none/Project/bisandro-rest/mobile/ReactComputerVisionTemplate/src/layermodel/model/model.json"
    );
    // console.log(net)

    // axios.get('http://localhost:1234').then(res=>{
    //   console.log(res);
    // });

    //  Loop and detect hands
    setInterval(() => {
      detect(net);
    }, 500);
    // detect(net);
  };

  const detect = async net => {
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // console.log(webcamRef.current.video.readyState)
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      // canvasRef.current.width = videoWidth;
      // canvasRef.current.height = videoHeight;

      // 4. TODO - Make Detections
      const img = tf.browser.fromPixels(video);
      // console.log(img)
      const resized = tf.image.resizeBilinear(img, [180, 180]);
      // resized.shape = [null, 640, 480, 3]
      // console.log(resized)
      const casted = resized.cast("int32");
      const expanded = casted.expandDims(0);
      // console.log(expanded)
      // expanded.shape = [null, 640, 480, 3]
      // const obj = await net.executeAsync(expanded);
      // const obj = await net.predict(expanded);
      // const obj = "ASJGD";
      const obj = net.predict(expanded);
      // console.log("agdshs")
      //   const model = tf.sequential({
      //     layers: [tf.layers.dense({units: 1, inputShape: [10]})]
      //  });
      //  const obj = model.predict(tf.ones([8, 10]), {batchSize: 4}).print();

      // console.log()
      obj.print();
      // obj.softmax().print()
      console.log(obj.arraySync()[0]);
      // obj.arraySync()[0].argMax().print()
      const v = tf.tensor(obj.dataSync());
      const vv = v.argMax().dataSync()[0];
      console.log(vv);
      console.log(className[vv]);
      console.log("SKIP-----------------------------------");
      setHasil([className[vv], obj.arraySync()[0][vv]]);

      // const boxes = await obj[1].array();
      // const classes = await obj[2].array();
      // const scores = await obj[4].array();

      // Draw mesh
      // const ctx = canvasRef.current.getContext("2d");

      // // 5. TODO - Update drawing utility
      // // drawSomething(obj, ctx)
      // requestAnimationFrame(() => {
      //   drawRect(
      //     boxes[0],
      //     classes[0],
      //     scores[0],
      //     0.8,
      //     videoWidth,
      //     videoHeight,
      //     ctx
      //   );
      // });

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
      <h5>
        {hasil[0]} ({hasil[1]})
      </h5>
      <Webcam
        ref={webcamRef}
        muted={true}
        videoConstraints={{ facingMode: { exact: "environment" } }}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: "90%"
          // top: 0
          // height: "40%"
        }}
      />
    </div>
  );
}

export default App;
