function downloadRankingCSV() {
  window.open("http://127.0.0.1:8000/download-ranking", "_blank")
}

export default function RankingTable({ ranking }) {
  return (
    <>
      {/* Header + Download */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold text-white">
          Ranking ({ranking.length} candidates)
        </h2>

        <button
          onClick={downloadRankingCSV}
          className="bg-emerald-600 hover:bg-emerald-700 px-4 py-2 rounded-lg text-sm font-semibold text-white"
        >
          Download Ranking CSV
        </button>
      </div>

      <div className="space-y-4">
        {ranking.map((r, i) => {
          const finalPercent = Math.round((r.final_score ?? 0) * 100)
          const skillPercent = Math.round((r.semantic_skills_score ?? 0) * 100)

          return (
            <div
              key={i}
              className="bg-white/5 border border-white/10 rounded-xl p-4"
            >
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-semibold text-white text-lg">
                    {r.person_name}
                  </p>
                  <p className="text-sm text-gray-400">
                    {r.email}
                  </p>

                  <div className="mt-2 text-sm text-gray-300 space-y-1">
                    <p>
                      Degree:{" "}
                      <span className="text-white">
                        {r.highest_degree || "Not specified"}
                      </span>
                    </p>

                    <p>
                      Skill Match:{" "}
                      <span className="text-white">
                        {skillPercent}%
                      </span>
                      <span className="text-gray-400 text-xs ml-2">
                        (semantic alignment with required skills)
                      </span>
                    </p>
                  </div>
                </div>

                {/* Final Score Bar */}
                <div className="w-48 text-right">
                  <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500"
                      style={{ width: `${finalPercent}%` }}
                    />
                  </div>

                  <p className="text-sm mt-2 text-white font-semibold">
                    Match Score: {finalPercent}%
                  </p>

                  <p className="text-xs text-gray-400">
                    (overall weighted ranking score)
                  </p>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}