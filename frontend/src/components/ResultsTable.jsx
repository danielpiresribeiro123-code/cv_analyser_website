function downloadCSV() {
  window.open("http://127.0.0.1:8000/download", "_blank")
}

export default function ResultsTable({ results }) {
  return (
    <>
      <h2 className="text-2xl font-semibold mb-4 text-center text-white">
        Results ({results.num_cvs} CVs)
      </h2>

      <div className="flex justify-end mb-4">
        <button
          onClick={downloadCSV}
          className="bg-emerald-600 hover:bg-emerald-700 px-4 py-2 rounded-lg text-sm font-semibold text-white"
        >
          Download CSV
        </button>
      </div>

      <div className="overflow-auto">
        <table className="w-full text-left border-collapse text-sm text-white">
          <thead>
            <tr className="border-b border-white/10 text-gray-400">
              <th className="py-2">Name</th>
              <th className="py-2">Email</th>
              <th className="py-2">Experience</th>
              <th className="py-2">Languages</th>
            </tr>
          </thead>

          <tbody>
            {results.results.map((r, i) => (
              <tr key={i} className="border-b border-white/5">
                <td className="py-2">{r["person name"]}</td>
                <td className="py-2">{r.email}</td>
                <td className="py-2">{r.experience_years ?? "-"}</td>
                <td className="py-2">{r.languages}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
