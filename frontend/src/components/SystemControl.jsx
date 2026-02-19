export default function SystemControl({ onReset }) {
  async function handleReset() {
    await fetch("http://127.0.0.1:8000/reset", {
      method: "POST",
    })

    onReset()
  }

  return (
    <div className="flex justify-center mt-4">
      <button
        onClick={handleReset}
        className="bg-red-600 hover:bg-red-700 px-6 py-2 rounded-xl font-semibold transition"
      >
        Reset System
      </button>
    </div>
  )
}
