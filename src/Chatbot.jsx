import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { Input, Button } from 'antd';
import axios from 'axios';
import './Chatbot.css';
import ReactMarkdown from 'react-markdown';

const { TextArea } = Input;

const Chatbot = () => {
  const location = useLocation();
  const disease = location.state?.disease || '';
  const initialMessage = location.state?.initialMessage || '';
  const chatListRef = useRef(null);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (initialMessage) {
      setMessages([{ role: 'system', content: initialMessage }]);
    }
  }, [disease, initialMessage]);

  useEffect(() => {
    if (chatListRef.current) {
      chatListRef.current.scrollTop = chatListRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const newMessage = { role: 'user', content: input };
    setMessages([...messages, newMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://127.0.0.1:5000/chat', {
        conversation: [...messages, newMessage]
      });

      const botMessage = response.data.content;
      setMessages([...messages, newMessage, { role: 'system', content: botMessage }]);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setLoading(false);
    }
  };
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className='App'>
      <header className="App-header">
        <a href='/'>
          <img src="public/Screen Shot 2025-03-01 at 6.30.25 AM.png" alt="logo" className='logo'/>
        </a>
      </header>
      
      <div className="chatbot-container">
        <h1 className='chatbot-title'>{disease}</h1>
        
        <div ref={chatListRef} className="chat-list">
          {messages.map((item, index) => (
            <div key={index} className={`${item.role}-message chat-message`}>
              <ReactMarkdown>{item.content}</ReactMarkdown>
            </div>
          ))}
        </div>

        <TextArea
          className="chat-input"
          rows={3}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask more about eco-friendly solutions..."
          disabled={loading}
        />
        <Button className='SbtBtn' type="primary" onClick={handleSend} loading={loading}>
          Send
        </Button>
      </div>
    </div>
  );
};

export default Chatbot;
