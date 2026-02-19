export default function UploadBox() {
  async function handleUpload(e) {
    const files = e.target.files
    if (!files.length) return

    const formData = new FormData()
    for (let f of files) formData.append("files", f)

    await fetch("http://127.0.0.1:8000/upload", {
      method: "POST",
      body: formData,
    })
  }

  return (
    <div className="w-full flex flex-col items-center gap-3">
      <input
        type="file"
        multiple
        accept=".pdf",".png", ".jpg", ".jpeg", ".bmp", ".tiff", "*.webp"
        onChange={handleUpload}
        className="text-sm text-gray-300
                   file:mr-4 file:py-2 file:px-4
                   file:rounded-lg file:border-0
                   file:text-sm file:font-semibold
                   file:bg-blue-600 file:text-white
                   hover:file:bg-blue-700"
      />
      <p className="text-gray-400 text-sm">
        Upload one or more PDF CVs
      </p>
    </div>
  )
}
