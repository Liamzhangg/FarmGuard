import React, { useState, useRef } from 'react';
import { Upload, Button, Image, Modal, Spin } from 'antd';
import { UploadOutlined, CameraOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom'; // ✅ Ensure it's inside the component
import Webcam from 'react-webcam';
import axios from 'axios';
import './App.css';
import { Routes, Route } from 'react-router-dom';
import Chatbot from './Chatbot';

const App = () => {
  const [fileList, setFileList] = useState([]);
  const [previewImage, setPreviewImage] = useState('');
  const [previewOpen, setPreviewOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [webcamOpen, setWebcamOpen] = useState(false);
  const webcamRef = useRef(null);
  const navigate = useNavigate(); // ✅ Ensure it's inside the component

  // Function to handle file upload
  const handleUpload = ({ fileList }) => {
    setFileList(fileList);
  };

  // Function to handle form submission
  const handleSubmit = async () => {
    if (fileList.length === 0) {
      console.error('No file selected');
      return;
    }

    const formData = new FormData();
    formData.append('file', fileList[0].originFileObj);

    setLoading(true);

    try {
      // Step 1: Predict the disease
      const predictResponse = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const predictedClass = predictResponse.data.predicted_class;
      console.log('🦠 Predicted Disease:', predictedClass);

      if (predictedClass.toLowerCase() === "not sure") {
        alert("The system is not sure about the disease. Please try again with a different image.");
        return;
      }

      const solutionResponse = await axios.post('http://127.0.0.1:5000/get_solution', {
        disease: predictedClass
      });

      const solution = solutionResponse.data.eco_friendly_solution;

      // Step 2: Redirect to chatbot page with the predicted class
      console.log('🔀 Redirecting to /chat...');
      navigate('/chat', { state: { disease: predictedClass, initialMessage: solution } });
      console.log('navigation executed');
      
    } catch (error) {
      console.error('❌ Error:', error);
    } finally {
      setLoading(false);
    }
  };

  // ✅ Define the missing capture function
  const capture = () => {
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    const file = dataURLtoFile(imageSrc, 'webcam.jpg');
    setFileList([{ uid: '-1', name: 'webcam.jpg', status: 'done', originFileObj: file }]);
    setWebcamOpen(false);
  };

  // ✅ Convert base64 to File object
  const dataURLtoFile = (dataurl, filename) => {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  return (
    <Routes>
      <Route path="/" element= {
      <div className="App">
      <header className="App-header">
        <a href='/'>
          <img src="public/Screen Shot 2025-03-01 at 6.30.25 AM.png" alt="logo" className='logo'/>
        </a>
      </header>

      <div className="container">
        <Upload
          listType="picture-card"
          fileList={fileList}
          onChange={handleUpload}
          beforeUpload={() => false} // Prevent automatic upload
        >
          {fileList.length >= 1 ? null : (
            <div>
              <UploadOutlined />
              <div style={{ marginTop: 8 }}>Upload</div>
            </div>
          )}
        </Upload>
        {loading && (
        <div className="loading-spinner">
          <Spin size="large" />
        </div>
      )}
        <Button className="SSbtBtn" type="primary" onClick={handleSubmit} style={{ marginTop: 16 }}>
          Submit
        </Button>

        {/* ✅ Camera Capture Feature */}
        <Button
          className="SSbtBtn"
          type="primary"
          icon={<CameraOutlined />}
          onClick={() => setWebcamOpen(true)}
          style={{ marginTop: 16 }}
        >
          Take Picture
        </Button>

        <Modal open={webcamOpen} onCancel={() => setWebcamOpen(false)} footer={null}>
          <Webcam audio={false} ref={webcamRef} screenshotFormat="image/jpeg" width="100%" />
          <Button type="primary" onClick={capture} style={{ marginTop: 16 }}>
            Capture
          </Button>
        </Modal>
      </div>
    </div> } />
    <Route path="/chat" element={<Chatbot />} />
    </Routes>
  );
};

export default App;
