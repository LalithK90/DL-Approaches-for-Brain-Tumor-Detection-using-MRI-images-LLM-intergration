import React from 'react';
import './Footer.css';

const Footer: React.FC = () => {
  return (
    <div className="footer-container">
      <div className="row justify-content-center mt-4">
        {/* Caution Section */}
        <div className="container-fluid bg-danger p-2">
          <p className="text-center text-white fw-bold mb-0">
            Disclaimer: This tool is for academic research purposes only and is <strong>not for medical diagnostic use.</strong>
          </p>
          <p className="text-center text-white small mb-0">
            The analysis is based on publicly available data and has not been clinically validated.
          </p>
        </div>
        {/* Accreditation Section */}
        <div className="container text-center mt-3">
          <p className="text-muted small mb-1">
            This is a final research work for the Master of Science in Computer Science (MSc in CS - SLQF Level 10) degree program conducted by the <a href="https://www.pgis.lk" target="_blank" rel="noopener noreferrer">Postgraduate Institute of Science (PGIS)</a> and the Department of Statistics & Computer Science, University of Peradeniya.
          </p>
          <p className="text-muted small mb-1">
            All rights reserved © PGIS & Department of Statistics & Computer Science, University of Peradeniya.
          </p>
          <p className="text-center small">
            <a href="https://github.com/LalithK90/Deep-Learning-Approaches-for-Brain-Tumor-Detection-using-MRI-WebApp.git">View Project on Github</a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Footer;