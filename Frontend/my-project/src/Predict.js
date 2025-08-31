import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function Predict() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-indigo-700 mb-6 text-center">
        Disease Identification
      </h1>

      {/* Upload Section */}
      <div className="bg-white p-6 rounded-2xl shadow-lg">
        <div className="flex flex-col items-center space-y-4">
          {preview && (
            <img
              src={preview}
              alt="Uploaded Preview"
              className="w-72 h-72 object-cover rounded-xl shadow-md"
            />
          )}
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="border border-gray-300 rounded-lg px-4 py-2 w-full cursor-pointer"
          />
          <button
            onClick={handleUpload}
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white px-6 py-2 rounded-xl shadow-lg transition duration-200"
          >
            {loading ? "Analyzing..." : "Predict"}
          </button>
        </div>
      </div>

      {/* Result Section */}
      {result && !result.error && (
        <div className="mt-8 bg-gray-50 p-6 rounded-xl shadow-inner">
          <h2 className="text-2xl font-semibold text-indigo-600 mb-4">
            Prediction: {result.prediction}
          </h2>

          {/* Confidence Chart */}
          {result.probabilities && (
            <div className="h-64 mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={result.probabilities}>
                  <XAxis dataKey="class" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Bar dataKey="confidence" fill="#6366F1" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Recommendation */}
          <p className="text-gray-700 whitespace-pre-line leading-relaxed">
            {result.recommendation}
          </p>
        </div>
      )}

      {/* Error */}
      {result && result.error && (
        <div className="mt-6 p-4 bg-red-100 text-red-700 rounded-lg">
          Error: {result.error}
        </div>
      )}
    </div>
  );
}
