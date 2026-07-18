import { useMemo } from 'react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

export const useApi = () => {
  const { token, logout } = useAuth();
  const navigate = useNavigate();

  const api = useMemo(() => {
    const instance = axios.create({
      baseURL: '/v1',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    instance.interceptors.request.use((config) => {
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    instance.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          logout();
          navigate('/login');
        } else if (error.response?.status === 429) {
          console.warn('Rate limit exceeded');
          // Add toast logic here later
        }
        return Promise.reject(error);
      }
    );

    return instance;
  }, [token, logout, navigate]);

  return api;
};
