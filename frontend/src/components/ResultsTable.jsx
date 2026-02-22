import { useState } from "react"
import { ChevronDown, ChevronUp } from "lucide-react"


function PhotoBlock({ photoUrl }) {
  const [hasError, setHasError] = useState(false)

  if (!photoUrl || hasError) {
    return (
      <div className="w-44 h-44 bg-white/5 border border-white/10 rounded-xl flex items-center justify-center text-gray-500 text-xs">
        No Photo
      </div>
    )
  }

  return (
    <img
      src={photoUrl}
      alt="CV Photo"
      className="w-44 h-44 object-cover rounded-xl border border-white/20"
      onError={() => setHasError(true)}
    />
  )
}

function downloadCSV() {
  window.open("http://127.0.0.1:8000/download", "_blank")
}

export default function ResultsTable({ results }) {
  const [openIndex, setOpenIndex] = useState(null)

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-semibold text-white">
          Results ({results.num_cvs} CVs)
        </h2>

        <button
          onClick={downloadCSV}
          className="bg-emerald-600 hover:bg-emerald-700 px-4 py-2 rounded-lg text-sm font-semibold text-white"
        >
          Download Results CSV
        </button>
      </div>

      <div className="space-y-4">
        {results.results.map((r, i) => {
          const isOpen = openIndex === i

          return (
            <div
              key={i}
              className="bg-white/5 border border-white/10 rounded-xl p-4"
            >
              {/* Header */}
              <div
                onClick={() =>
                  setOpenIndex(isOpen ? null : i)
                }
                className="flex justify-between items-center cursor-pointer"
              >
                <div>
                  <p className="font-semibold text-white text-lg">
                    {r["person name"]}
                  </p>
                  <p className="text-sm text-gray-400">
                    {r.email}
                  </p>
                </div>

                <div className="flex items-center gap-4">
                  <span className="text-sm text-gray-300">
                    {r.experience_years} years
                  </span>

                  {isOpen ? (
                    <ChevronUp />
                  ) : (
                    <ChevronDown />
                  )}
                </div>
              </div>

              {/* Expanded content */}
              {isOpen && (
                <div className="mt-4 grid md:grid-cols-3 gap-6 text-sm text-gray-300">
                  {/* Left column */}
                  <div className="space-y-2">
                    <p>
                      <span className="text-white">Degree:</span>{" "}
                      {r.degree || "-"}
                    </p>
                    <p>
                      <span className="text-white">Company:</span>{" "}
                      {r.company || "-"}
                    </p>
                    <p>
                      <span className="text-white">University:</span>{" "}
                      {r.university || "-"}
                    </p>
                    <p>
                      <span className="text-white">Languages:</span>{" "}
                      {r.languages}
                    </p>
                  </div>

                  {/* Middle column */}
                  <div>
                    <p className="text-white mb-1">Skills:</p>
                    <div className="flex flex-wrap gap-2">
                      {r.skills
                        ?.split(",")
                        .slice(0, 10)
                        .map((skill, idx) => (
                          <span
                            key={idx}
                            className="bg-blue-600/30 px-2 py-1 rounded text-xs"
                          >
                            {skill.trim()}
                          </span>
                        ))}
                    </div>
                  </div>

                  {/* Photo */}
                  <div className="flex justify-center items-start pt-0 -mt-1">
                    <PhotoBlock photoUrl={r.photo_url} />
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </>
  )
}