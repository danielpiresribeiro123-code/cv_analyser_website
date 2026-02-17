export default function RankingTable({ ranking }) {
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 mt-10">
      <h2 className="text-2xl font-semibold mb-4 text-center">
        Top Candidates ({ranking.num_cvs})
      </h2>

      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="border-b border-gray-700 text-gray-400">
            <th className="py-2">Rank</th>
            <th className="py-2">File</th>
            <th className="py-2">Semantic</th>
            <th className="py-2">Experience (years)</th>
            <th className="py-2">Final Score</th>
          </tr>
        </thead>

        <tbody>
          {ranking.ranking.map((r, i) => (
            <tr key={i} className="border-b border-gray-800">
              <td className="py-2 font-semibold text-purple-400">#{i + 1}</td>
              <td className="py-2">{r.file}</td>
              <td className="py-2">
                {(r.semantic_score * 100).toFixed(1)}%
              </td>
              <td className="py-2">{r.experience_years.toFixed(1)}</td>
              <td className="py-2 font-semibold text-green-400">
                {(r.final_score * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
