import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(() => localStorage.getItem('ecopackai_token'));
  const [user, setUser] = useState(null);

  useEffect(() => {
    if (token) {
      try {
        const payload = token.split('.')[1];
        const decoded = JSON.parse(atob(payload));
        setUser({
          name: decoded.name,
          email: decoded.email,
          roles: decoded.roles,
        });
        localStorage.setItem('ecopackai_token', token);
      } catch (err) {
        console.error('Failed to decode token:', err);
        logout();
      }
    } else {
      setUser(null);
      localStorage.removeItem('ecopackai_token');
    }
  }, [token]);

  const login = (newToken) => {
    setToken(newToken);
  };

  const logout = () => {
    setToken(null);
    window.location.href = '/login';
  };

  const isAuthenticated = !!token && !!user;

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isAuthenticated }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
