function downloadCSV() {
  window.open("http://127.0.0.1:8000/download-results", "_blank")
}

export default function ResultsTable({ results }) {
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6">
      <h2 className="text-2xl font-semibold mb-4 text-center">
        Results ({results.num_cvs} CVs)
      </h2>
      <div className="flex justify-end mb-4">
        <button
          onClick={downloadCSV}
          className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg text-sm font-semibold"
        >
          Download CSV
        </button>
      </div>

      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-gray-700 text-gray-400">
            <th className="py-2">File</th>
            <th className="py-2">Predicted Role</th>
            <th className="py-2">Score</th>
          </tr>
        </thead>

        <tbody>
          {results.results.map((r, i) => (
            <tr key={i} className="border-b border-gray-800">
              <td className="py-2">{r.file}</td>
              <td className="py-2">{r.label}</td>
              <td className="py-2 text-blue-400 font-semibold">
                {(r.score * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
