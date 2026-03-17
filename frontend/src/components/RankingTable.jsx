import { useState } from "react"
import { ChevronDown, ChevronUp } from "lucide-react"

function clamp01(x) {
  if (x == null || isNaN(x)) return 0
  return Math.min(1, Math.max(0, x))
}

function toPercent(x) {
  return Math.round(clamp01(x) * 100)
}

function downloadRankingCSV() {
  window.open("http://127.0.0.1:8000/download-ranking", "_blank")
}

export default function RankingTable({ ranking, weights }) {
  const [openIndex, setOpenIndex] = useState(null)

  if (!ranking || !Array.isArray(ranking)) return null

  const normalizedWeights = weights
    ? Object.fromEntries(
        Object.entries(weights).map(([k, v]) => [k, Number(v) / 100])
      )
    : {
        semantic_raw: 0.27,
        semantic_skills: 0.30,
        exact_skills: 0.03,
        semantic_job_title: 0.08,
        experience_general: 0.02,
        experience_specific: 0.14,
        degree: 0.05,
        language: 0.06,
        document_length: 0.03,
        photo: 0.02,
      }

  return (
    <>
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
            const isOpen = openIndex === i
            const finalPercent = toPercent(r?.final_score ?? 0)

            const metrics = [
              {
                label: "Skills (semantic)",
                value: clamp01(r?.semantic_skills_score),
                weight: normalizedWeights.semantic_skills,
              },
              {
                label: "Skills (exact)",
                value: clamp01(r?.exact_skill_score),
                weight: normalizedWeights.exact_skills,
              },
              {
                label: "Profile Fit",
                value: clamp01(r?.semantic_raw_score),
                weight: normalizedWeights.semantic_raw,
              },
              {
                label: "Job Title Fit",
                value: clamp01(r?.semantic_job_title_score),
                weight: normalizedWeights.semantic_job_title,
              },
              {
                label: "Experience (general)",
                value: clamp01(r?.experience_score_general),
                weight: normalizedWeights.experience_general,
                extra: `${r?.experience_years ?? 0} years`,
              },
              {
                label: "Experience (specific)",
                value: clamp01(r?.experience_score_specific),
                weight: normalizedWeights.experience_specific,
              },
              {
                label: "Education",
                value: clamp01(r?.degree_score),
                weight: normalizedWeights.degree,
                extra: r?.highest_degree,
              },
              {
                label: "Languages",
                value: clamp01(r?.language_score),
                weight: normalizedWeights.language,
              },
              {
                label: "Document Length",
                value: clamp01(r?.document_length_score),
                weight: normalizedWeights.document_length,
                extra: `${r?.ocr_text_length ?? 0} text / ${r?.num_text_blocks ?? 0} blocks`,
              },
              {
                label: "Photo Presence",
                value: clamp01(r?.photo_score),
                weight: normalizedWeights.photo,
                extra: `${r?.num_photos ?? 0} photo(s)`,
              },
            ]

          return (
            <div
              key={i}
              className="bg-white/5 border border-white/10 rounded-xl p-4"
            >
              {/* Header */}
              <div
                onClick={() => setOpenIndex(isOpen ? null : i)}
                className="flex justify-between items-center cursor-pointer"
              >
                <div>
                  <p className="font-semibold text-white text-lg">
                    {r?.person_name}
                  </p>
                  <p className="text-sm text-gray-400">
                    {r?.email}
                  </p>
                </div>

                <div className="flex items-center pt-2 gap-4">
                  <div className="w-52 text-right">
                    <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-500"
                        style={{ width: `${finalPercent}%` }}
                      />
                    </div>

                    <p className="text-sm mt-2 text-white font-semibold">
                      Match Score: {finalPercent}%
                    </p>
                  </div>

                  {isOpen ? <ChevronUp /> : <ChevronDown />}
                </div>
              </div>

              {/* Expanded content */}
              {isOpen && (
                <div className="mt-6 grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {metrics.map((m, idx) => {
                    const valuePct = toPercent(m.value)
                    const weightPct = Math.round(m.weight * 100)
                    const contribution = Math.round(clamp01(m.value) * m.weight * 100)

                    return (
                      <div
                        key={idx}
                        className="bg-white/5 border border-white/10 rounded-lg p-4 flex flex-col items-center text-center min-h-[140px]"
                      >
                        <p className="text-white font-semibold">
                          {m.label}
                        </p>

                        <p className="text-gray-300 mt-2">
                          {valuePct}%
                        </p>

                        <p className="text-gray-400 text-xs mt-1">
                          Weight {weightPct}%
                        </p>

                        <p className="text-gray-400 text-xs">
                          Contribution ~{contribution}%
                        </p>

                        {m.extra && (
                          <p className="text-xs text-gray-500 mt-2">
                            {m.extra}
                          </p>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </>
  )
}