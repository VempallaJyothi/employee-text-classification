function ResultCard({ result }) {
  const percent = Math.round(result.confidence * 100);

  return (
    <section className="result">
      <h2>Result</h2>
      <p>
        <strong>Category:</strong> {result.category}
      </p>
      <p>
        <strong>Confidence:</strong> {percent}%
      </p>
      <div className="bar">
        <div className="bar-fill" style={{ width: `${percent}%` }} />
      </div>
      <p className="original">
        <strong>Original text:</strong> {result.text}
      </p>
    </section>
  );
}

export default ResultCard;
