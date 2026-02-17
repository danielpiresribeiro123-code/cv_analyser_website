export default function UploadBox() {
  async function handleUpload(e) {
    const files = e.target.files
    if (!files.length) return

    const formData = new FormData()
    for (let f of files) formData.append("files", f)

    await fetch("http://127.0.0.1:8000/upload-cvs", {
      method: "POST",
      body: formData,
    })

    alert("CVs uploaded successfully!")
  }

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 text-center">
      <input
        type="file"
        multiple
        accept=".pdf"
        onChange={handleUpload}
        className="block mx-auto text-sm"
      />
      <p className="text-gray-400 mt-3">Upload one or more PDF CVs</p>
    </div>
  )
}
