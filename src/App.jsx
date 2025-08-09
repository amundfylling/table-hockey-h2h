import React from 'react';
import { DataProvider } from './context/DataContext';
import Dashboard from './pages/Dashboard';

function App() {
  return (
    <DataProvider>
      <Dashboard />
    </DataProvider>
  );
}

export default App;
