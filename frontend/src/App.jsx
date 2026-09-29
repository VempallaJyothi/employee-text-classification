import { useState } from "react";
import ResultCard from "./components/ResultCard";
import "./App.css";

// The address of our FastAPI endpoint
const API_URL = "http://127.0.0.1:8000/predict";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleClassify() {
    if (!text.trim()) {
      setError("Please enter some text first.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      // Send the text to FastAPI as JSON
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text }),
      });

      // The server answered, but with an error status (400, 422, 500...)
      if (!response.ok) {
        const data = await response.json().catch(() => null);
        let message = "Something went wrong.";
        if (data && typeof data.detail === "string") {
          message = data.detail;
        } else if (data && Array.isArray(data.detail) && data.detail[0]?.msg) {
          message = data.detail[0].msg;
        }
        throw new Error(message);
      }

      // Success: read the JSON and show it
      const data = await response.json();
      setResult(data);
    } catch (err) {
      // fetch() throws a TypeError when the server cannot be reached at all
      if (err instanceof TypeError) {
        setError(
          "Cannot reach the server. Is the FastAPI backend running at http://127.0.0.1:8000 ?"
        );
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>Employee Text Classification</h1>
      <p className="subtitle">
        Enter employee skills or a description and the model will predict the
        category.
      </p>

      <textarea
        rows="5"
        placeholder="Example: I am a Java developer with Spring Boot experience."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button onClick={handleClassify} disabled={loading}>
        {loading ? "Classifying..." : "Classify"}
      </button>

      {error && <div className="error">{error}</div>}

      {result && <ResultCard result={result} />}
    </main>
  );
}

export default App;
