// filepath: /Users/Ferdinand/FarmGuard-1/src/main.jsx
import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import 'antd/dist/reset.css';  // New CSS import for Ant Design 5+

const container = document.getElementById('root');
const root = createRoot(container);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);