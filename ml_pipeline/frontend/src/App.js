import { useState } from 'react';
import './App.css';

const API = 'http://localhost:8000';

const initialForm = { age: '', income: '', score: '', category: 'A', region: 'North' };

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState('');

  const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError('');
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          age: parseInt(form.age),
          income: parseInt(form.income),
          score: parseFloat(form.score),
          category: form.category,
          region: form.region,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail);
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    setRetraining(true);
    setRetrainMsg('');
    try {
      const res = await fetch(`${API}/retrain`, { method: 'POST' });
      const data = await res.json();
      setRetrainMsg(`Retrained! Accuracy: ${data.metrics.accuracy} | ROC-AUC: ${data.metrics.roc_auc}`);
    } catch {
      setRetrainMsg('Retrain failed.');
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>ML Prediction Pipeline</h1>
        <p>Enter features below to get a prediction</p>
      </header>

      <main className="main">
        <div className="card">
          <form onSubmit={handlePredict} className="form">
            <div className="form-grid">
              <div className="field">
                <label>Age (18–80)</label>
                <input type="number" name="age" min="18" max="80" value={form.age}
                  onChange={handleChange} required placeholder="e.g. 35" />
              </div>
              <div className="field">
                <label>Income (20k–200k)</label>
                <input type="number" name="income" min="20000" max="200000" value={form.income}
                  onChange={handleChange} required placeholder="e.g. 75000" />
              </div>
              <div className="field">
                <label>Score</label>
                <input type="number" name="score" step="0.01" value={form.score}
                  onChange={handleChange} required placeholder="e.g. 0.82" />
              </div>
              <div className="field">
                <label>Category</label>
                <select name="category" value={form.category} onChange={handleChange}>
                  {['A', 'B', 'C'].map(c => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div className="field">
                <label>Region</label>
                <select name="region" value={form.region} onChange={handleChange}>
                  {['North', 'South', 'East', 'West'].map(r => <option key={r}>{r}</option>)}
                </select>
              </div>
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </form>

          {error && <div className="alert error">{error}</div>}

          {result && (
            <div className={`result ${result.prediction === 1 ? 'positive' : 'negative'}`}>
              <h2>Result: {result.label}</h2>
              <p>Prediction: <strong>{result.prediction}</strong></p>
              <p>Probability: <strong>{(result.probability * 100).toFixed(2)}%</strong></p>
              <div className="prob-bar">
                <div className="prob-fill" style={{ width: `${result.probability * 100}%` }} />
              </div>
            </div>
          )}
        </div>

        <div className="card retrain-card">
          <h3>Model Management</h3>
          <p>Retrain the model on fresh data and reload automatically.</p>
          <button className="btn-secondary" onClick={handleRetrain} disabled={retraining}>
            {retraining ? 'Retraining...' : 'Retrain Model'}
          </button>
          {retrainMsg && <div className="alert info">{retrainMsg}</div>}
        </div>
      </main>
    </div>
  );
}

export default App;
