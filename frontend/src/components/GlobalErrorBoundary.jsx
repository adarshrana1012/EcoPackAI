import React from 'react';

export class GlobalErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-brand-dark flex flex-col items-center justify-center text-brand-light p-6">
          <div className="bg-brand-dark border border-brand-red p-8 rounded-xl shadow-2xl max-w-md w-full text-center">
            <h1 className="text-3xl font-bold text-brand-red mb-4">Oops!</h1>
            <p className="text-lg mb-6">Something went wrong.</p>
            <p className="text-sm text-gray-400 mb-8 font-mono bg-black p-4 rounded text-left overflow-auto">
              {this.state.error?.message || 'Unknown error'}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2 bg-brand-green hover:bg-emerald-600 text-white rounded-lg transition-colors"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
