export default function RankingTable({ ranking }) {
  return (
    <div className="space-y-3">
      {ranking.map((r, i) => (
        <div
          key={i}
          className="flex justify-between items-center
                     bg-white/5 border border-white/10
                     rounded-xl px-4 py-3"
        >
          <div>
            <p className="font-semibold text-white">{r.name}</p>
            <p className="text-sm text-gray-400">{r.email}</p>
          </div>

          <div className="w-40">
            <div className="h-2 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500"
                style={{ width: `${r.score}%` }}
              />
            </div>
            <p className="text-xs text-right mt-1 text-gray-400">
              {r.score}%
            </p>
          </div>
        </div>
      ))}
    </div>
  )
}
