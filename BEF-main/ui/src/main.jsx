import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './style.css'
import { EngineProvider } from './state/EngineContext.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <EngineProvider>
        <App />
      </EngineProvider>
    </BrowserRouter>
  </React.StrictMode>,
)
