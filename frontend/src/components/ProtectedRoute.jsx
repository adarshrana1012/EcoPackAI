import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export const ProtectedRoute = ({ requiredRole }) => {
  const { user, isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (requiredRole && (!user?.roles || !user.roles.includes(requiredRole))) {
    return <Navigate to="/" replace />; // or an unauthorized page
  }

  return <Outlet />;
};
