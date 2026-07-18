import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useApi } from './useApi';

/**
 * Custom hook returning a useMutation wrapper for product fragility classification.
 * Caches the classification result in the queryClient under ['classification', inputData].
 */
export const useClassifyProduct = () => {
  const api = useApi();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (productData) => {
      const response = await api.post('/classify', productData);
      return response.data;
    },
    onSuccess: (data, variables) => {
      queryClient.setQueryData(['classification', variables], data);
    },
  });
};

/**
 * Custom hook returning a useMutation wrapper for order bin packing optimization.
 * Invalidates ['analytics'] on success to force operational metrics updates.
 */
export const usePackOrder = () => {
  const api = useApi();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (orderData) => {
      const response = await api.post('/pack', orderData);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analytics'] });
      queryClient.invalidateQueries({ queryKey: ['shipments'] });
    },
  });
};

/**
 * Custom query hook for getting aggregate dashboard metrics.
 */
export const useAnalytics = (startDate, endDate) => {
  const api = useApi();

  return useQuery({
    queryKey: ['analytics', startDate, endDate],
    queryFn: async () => {
      const params = {};
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;
      const response = await api.get('/metrics/aggregate', { params });
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
  });
};

/**
 * Custom query hook for retrieving registered ML model versions.
 */
export const useModelVersions = () => {
  const api = useApi();

  return useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await api.get('/models/versions');
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
  });
};

/**
 * Custom query hook for checking service gateway liveness and Redis health.
 */
export const useSystemHealth = () => {
  const api = useApi();

  return useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const response = await api.get('/gateway/health');
      return response.data;
    },
    refetchInterval: 30000, // poll every 30 seconds
    staleTime: 0,
  });
};

/**
 * Custom query hook for paginated shipments.
 */
export const useShipments = (filters) => {
  const api = useApi();

  return useQuery({
    queryKey: ['shipments', filters],
    queryFn: async () => {
      const response = await api.get('/shipments', { params: filters });
      return response.data;
    },
    staleTime: 60 * 1000, // 1 minute
    keepPreviousData: true,
  });
};
