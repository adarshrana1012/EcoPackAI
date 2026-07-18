import { useState } from 'react';
import { useApi } from './useApi';

export const usePackOrder = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const api = useApi();

  const pack = async (orderData) => {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await api.post('/pack', orderData);
      setResult(response.data);
      return response.data;
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Packing request failed.');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { pack, loading, error, result };
};
