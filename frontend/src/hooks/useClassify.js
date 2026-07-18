import { useState } from 'react';
import { useApi } from './useApi';

export const useClassify = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const api = useApi();

  const classify = async (features) => {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await api.post('/classify', features);
      setResult(response.data);
      return response.data;
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Classification request failed.');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { classify, loading, error, result };
};
