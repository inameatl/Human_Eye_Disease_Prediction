export default function About() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-indigo-700 mb-4">About the Project</h1>
      <p className="text-gray-700 leading-relaxed mb-4">
        Retinal OCT captures high-resolution cross sections of the retina. 
        Our dataset contains <strong>84,495 images</strong> categorized into CNV, DME, Drusen, 
        and Normal — verified by multiple expert graders and ophthalmologists.
      </p>

      <h2 className="text-xl font-semibold text-indigo-600 mb-2">Dataset Sources</h2>
      <ul className="list-disc pl-6 text-gray-700 space-y-1">
        <li>Shiley Eye Institute</li>
        <li>California Retinal Research Foundation</li>
        <li>Shanghai First People’s Hospital</li>
        <li>Beijing Tongren Eye Center</li>
      </ul>

      <p className="mt-4 text-gray-600">
        Each image was validated through a tiered process (graders → ophthalmologists → senior retinal specialists) 
        to ensure accuracy.
      </p>
    </div>
  );
}
