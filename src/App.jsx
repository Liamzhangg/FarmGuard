import React, { useState } from 'react';
import { Upload, Button, Image } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import axios from 'axios';
import './App.css';

const App = () => {
  const [fileList, setFileList] = useState([]);
  const [previewImage, setPreviewImage] = useState('');
  const [previewOpen, setPreviewOpen] = useState(false);

  const handleUpload = ({ fileList }) => {
    setFileList(fileList);
  };

  const handleSubmit = async () => {
    if (fileList.length === 0) {
      alert('Please upload a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', fileList[0].originFileObj);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      alert(`Prediction: ${response.data.predicted_class}`);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please try again.');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="title">FarmGuard</h1>
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

        {previewImage && (
          <Image
            wrapperStyle={{ display: 'none' }}
            preview={{
              visible: previewOpen,
              onVisibleChange: (visible) => setPreviewOpen(visible),
              afterOpenChange: (visible) => !visible && setPreviewImage(''),
            }}
            src={previewImage}
          />
        )}

        <Button className="SbtBtn" type="primary" onClick={handleSubmit}>
          Submit
        </Button>
      </div>
    </div>
  );
};

export default App;
