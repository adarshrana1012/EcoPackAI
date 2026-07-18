import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { ProtectedRoute } from './components/ProtectedRoute';
import { GlobalErrorBoundary } from './components/GlobalErrorBoundary';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { Login } from './pages/Login';
import { Dashboard } from './pages/Dashboard';
import { Classify } from './pages/Classify';
import { PackOrder } from './pages/PackOrder';
import { Analytics } from './pages/Analytics';
import { Admin } from './pages/Admin';

import { ShipmentHistory } from './pages/ShipmentHistory';

const Layout = () => {
  return (
    <div className="flex flex-col h-screen w-screen bg-slate-50 dark:bg-slate-950 transition-colors duration-300">
      {/* Top Navbar */}
      <Navbar />

      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar />

        {/* Dynamic Nested Page view */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

function App() {
  return (
    <GlobalErrorBoundary>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            {/* Public Login Route */}
            <Route path="/login" element={<Login />} />

            {/* Protected Routes inside Layout */}
            <Route element={<ProtectedRoute />}>
              <Route element={<Layout />}>
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/classify" element={<Classify />} />
                <Route path="/pack" element={<PackOrder />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/shipments" element={<ShipmentHistory />} />
                
                {/* Admin-only route */}
                <Route element={<ProtectedRoute requiredRole="admin" />}>
                  <Route path="/admin" element={<Admin />} />
                </Route>
              </Route>
            </Route>

            {/* Fallback Redirects */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </GlobalErrorBoundary>
  );
}

export default App;
