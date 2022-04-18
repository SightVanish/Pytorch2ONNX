using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
public class ONNX_Unity : MonoBehaviour
{
    public NNModel onnxAsset;
    public Texture2D[] imageToRecognises;

    private IWorker worker;
    void Start()
    {
        worker = onnxAsset.CreateWorker();
    }

    public void inference()
    {
        foreach (var imageToRecognise in imageToRecognises)
        {
            // convert texture into Tensor of shape [1, imageToRecognise.height, imageToRecognise.width, 3]
            using (var input = new Tensor(imageToRecognise, channels: 1))
            {
                // execute neural network with specific input and get results back
                var output = worker.Execute(input).PeekOutput();

                // the following line will access values of the output tensor causing the main thread to block until neural network execution is done
                var indexWithHighestProbability = output.ArgMax()[0];

                UnityEngine.Debug.Log($"Image was recognised as class number: {indexWithHighestProbability}");
            }
        }
    }

    void OnDisable()
    {
        worker.Dispose();
    }
}

