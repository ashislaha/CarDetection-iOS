//
//  ViewController.swift
//  CarDetectionUsingML
//
//  Created by Ashis Laha on 29/07/17.
//  Copyright © 2017 Ashis Laha. All rights reserved.
//

import UIKit
import CoreML
import Vision

enum ButtonAction {
    case takePhoto
    case identifyPhoto
}

enum Classifier {
    case DNN
    case CNN
}

class BaseViewController: UIViewController , UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    let row = 100
    let column = 100
    let classes = 2
    
    var actionType : ButtonAction = .takePhoto {
        didSet {
            switch actionType {
            case .takePhoto : buttonOutlet.setTitle("Take Photo", for: .normal)
            case .identifyPhoto : buttonOutlet.setTitle("Identify Photo", for: .normal)
            }
        }
    }
    var classifier : Classifier = .CNN {
        didSet {
            switch classifier {
            case .CNN : title = "CNN"
            case .DNN : title = "DNN"
            }
        }
    }
    
    @IBAction func chooseClassifier(_ sender: UIBarButtonItem) {
        let actionSheet = UIAlertController(title: "Classfier", message: "", preferredStyle: .actionSheet)
        let dnn = UIAlertAction(title: "DNN", style: .default) { [weak self] (action) in
            self?.classifier = .DNN
        }
        let cnn = UIAlertAction(title: "CNN (Default)", style: .default) { [weak self] (action) in
            self?.classifier = .CNN
        }
        let cancel = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        actionSheet.addAction(cnn)
        actionSheet.addAction(dnn)
        actionSheet.addAction(cancel)
        self.present(actionSheet, animated: true, completion: nil)
    }
    
    @IBOutlet weak var imageView: UIImageView! {
        didSet {
            imageView.isHidden = true
        }
    }
    
    @IBOutlet weak var buttonOutlet: UIButton! {
        didSet {
            buttonOutlet.setTitle("Take Photo", for: .normal)
            buttonOutlet.backgroundColor = UIColor.brown.withAlphaComponent(0.5)
        }
    }
    
    @IBOutlet weak var label: UILabel! {
        didSet {
            label.isHidden = true
        }
    }
    
    @IBAction func buttonAction(_ sender: UIButton) {
        switch actionType {
        case .takePhoto : takePhoto()
        case .identifyPhoto: classify(image: imageView.image)
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    private func takePhoto() {
        label.text = ""
        let imagePickerVC = UIImagePickerController()
        imagePickerVC.delegate = self
        
        let actionSheet = UIAlertController(title: "Take Photo", message: "Car Detection", preferredStyle: .actionSheet)
        let cameraAction = UIAlertAction(title: "Camera", style: .default) { [weak self] (action) in
            imagePickerVC.sourceType = .camera
            self?.present(imagePickerVC, animated: true, completion: nil)
        }
        actionSheet.addAction(cameraAction)
        
        let photoLibrary = UIAlertAction(title: "Photo Galary", style: .default) { [weak self] (action) in
            imagePickerVC.sourceType = .photoLibrary
            self?.present(imagePickerVC, animated: true, completion: nil)
        }
        actionSheet.addAction(photoLibrary)
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        actionSheet.addAction(cancelAction)
        self.present(actionSheet, animated: true, completion: nil)
    }
    
    // Delegate method
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        if let image = info[UIImagePickerControllerOriginalImage] as? UIImage {
            imageView.image = image
            imageView.isHidden = false
            actionType = .identifyPhoto
        }
        picker.dismiss(animated: true, completion: nil)
    }
    
    
    //MARK:- Classification
    
    private func classify(image : UIImage?) {
        
        guard let image = image else { return }
        let imageInfo : (image : UIImage, pixel: [Double]) = ImagePreProcessing.shared.preProcessImage(image: image)
        
        imageView.image = imageInfo.image
        label.isHidden = false

        if classifier == .DNN {
            let mlModel = car_detection_keras_DNN()    // Input Matrix is [10000] Matrix - 1D Matrix
            guard let inputMatrix = try? MLMultiArray(shape: [10000], dataType: .double) else { fatalError("Unexpected runtime error. MLMultiArray") }
            
            // Feed data to inputMatrix
            for i in 0..<row*column { inputMatrix[i] = NSNumber(value: imageInfo.pixel[i]) }
            
            if let prediction = try? mlModel.prediction(input1: inputMatrix) {
                let outputs = prediction.output1
                print(outputs)
                var outputArray = [Double]()
                for i in 0..<classes { outputArray.append(Double(truncating: outputs[i])) }
                
                label.text = outputArray[0] > outputArray[1] ?  "It's A CAR" : "It's not a CAR"
            }
        } else if self.classifier == .CNN {
            let mlModel = car_detection_keras_CNN() // Input Matrix is [1, 100, 100] Matrix - 1D Matrix
            
            guard let inputMatrix = try? MLMultiArray(shape: [1,100,100], dataType: .double) else { fatalError("Unexpected runtime error. MLMultiArray") }
            
            // Feed data to inputMatrix
            for i in 0..<row*column { inputMatrix[i] = NSNumber(value: imageInfo.pixel[i]) }
            
            if let prediction = try? mlModel.prediction(input1: inputMatrix) {
                let outputs = prediction.output1
                print(outputs)
                var outputArray = [Double]()
                for i in 0..<classes { outputArray.append(Double(truncating: outputs[i])) }
                
                label.text = outputArray[0] > outputArray[1] ?  "It's A CAR" : "It's not a CAR"
            }
        }
        

        actionType = .takePhoto
    }
    
    
    //MARK:- Vision
    private func recognizeUsingVision() {
        let coreMLmodel = car_detection_keras_DNN()
        let model = try? VNCoreMLModel(for:coreMLmodel.model)
        let request = VNCoreMLRequest(model: model!, completionHandler: myResultsMethod)
        if let cgImage = imageView.image?.cgImage {
            let handler = VNImageRequestHandler(cgImage:cgImage, options: [:] )
            try? handler.perform([request])
        }
    }
    
    private func myResultsMethod(request: VNRequest, error: Error?) {
        guard let results = request.results as? [VNClassificationObservation] else { fatalError("Error in Results") }
        for classification in results {
            if classification.confidence > 0.25 {
                print(classification.identifier, classification.confidence)
                label.text = classification.identifier
            }
        }
    }
}


