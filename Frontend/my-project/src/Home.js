export default function Home() {
  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-indigo-700 mb-4">
        OCT Retinal Analysis Platform
      </h1>
      <p className="text-gray-700 leading-relaxed mb-6">
        Optical Coherence Tomography (OCT) is a powerful imaging technique that 
        provides high-resolution cross-sectional images of the retina, allowing 
        for early detection and monitoring of retinal diseases.
      </p>

      <h2 className="text-xl font-semibold text-indigo-600 mt-6 mb-2">
        Key Features
      </h2>
      <ul className="list-disc pl-6 text-gray-700 space-y-1">
        <li>Automated classification of OCT images into CNV, DME, Drusen, Normal</li>
        <li>Streamlined workflow: upload, analyze, review scans</li>
        <li>Trusted dataset with expert-verified labels</li>
      </ul>

      <h2 className="text-xl font-semibold text-indigo-600 mt-6 mb-2">
        Retinal Diseases
      </h2>
      <div className="grid md:grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 rounded-xl shadow">
          <h3 className="font-bold">CNV</h3>
          <p>Choroidal Neovascularization: subretinal fluid & neovascular membrane.</p>
        </div>
        <div className="p-4 bg-gray-50 rounded-xl shadow">
          <h3 className="font-bold">DME</h3>
          <p>Diabetic Macular Edema: retinal thickening & intraretinal fluid.</p>
        </div>
        <div className="p-4 bg-gray-50 rounded-xl shadow">
          <h3 className="font-bold">Drusen</h3>
          <p>Early AMD: presence of drusen deposits.</p>
        </div>
        <div className="p-4 bg-gray-50 rounded-xl shadow">
          <h3 className="font-bold">Normal</h3>
          <p>Preserved foveal contour without fluid.</p>
        </div>
      </div>
    </div>
  );
}
