// Global variables
let currentUser = null;
let cameras = [];
let accidents = [];

// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap modals
    const loginModal = new bootstrap.Modal('#loginModal');
    const registerModal = new bootstrap.Modal('#registerModal');
    const cameraModal = new bootstrap.Modal('#cameraModal');
    const videoModal = new bootstrap.Modal('#videoModal');

    // Check authentication status
    checkAuthStatus();
    
    // Setup all event listeners
    setupEventListeners();

    // Load sample data
    loadSampleData();

    function checkAuthStatus() {
        const token = localStorage.getItem('authToken');
        if (!token) {
            showLoginModal();
        } else {
            currentUser = JSON.parse(localStorage.getItem('currentUser')) || {name: "Admin"};
            showView('dashboard');
        }
    }

    function setupEventListeners() {
        // Navigation
        document.getElementById('dashboardLink').addEventListener('click', () => showView('dashboard'));
        document.getElementById('camerasLink').addEventListener('click', () => showView('cameras'));
        document.getElementById('accidentsLink').addEventListener('click', () => showView('accidents'));
        document.getElementById('settingsLink').addEventListener('click', () => showView('settings'));

        // Authentication
        document.getElementById('logoutBtn').addEventListener('click', logout);
        document.getElementById('showRegister').addEventListener('click', () => {
            loginModal.hide();
            registerModal.show();
        });
        document.getElementById('showLogin').addEventListener('click', () => {
            registerModal.hide();
            loginModal.show();
        });

        // Forms
        document.getElementById('loginForm').addEventListener('submit', handleLogin);
        document.getElementById('registerForm').addEventListener('submit', handleRegister);
        document.getElementById('alertSettingsForm').addEventListener('submit', saveSettings);

        // Camera management
        document.getElementById('addCameraBtn').addEventListener('click', () => {
            document.getElementById('cameraForm').reset();
            document.getElementById('cameraModalTitle').textContent = 'Add Camera';
            cameraModal.show();
        });
        document.getElementById('saveCameraBtn').addEventListener('click', saveCamera);

        // Video management
        document.getElementById('downloadVideoBtn').addEventListener('click', downloadVideo);
        document.getElementById('searchVideosBtn').addEventListener('click', searchVideos);
    }

    function loadSampleData() {
        // Sample camera data
        cameras = [
            { id: 1, name: "Main Street Camera", location: "Main Street Intersection", 
              url: "rtsp://camera1.example.com", username: "admin", password: "password123" },
            { id: 2, name: "Highway Camera", location: "Highway Exit 42", 
              url: "rtsp://camera2.example.com", username: "admin", password: "password123" }
        ];

        // Sample accident data
        accidents = [
            { id: 1, timestamp: new Date(), location: "Main Street Intersection", 
              cameraId: "CAM-001", severity: "High", videoUrl: "sample1.mp4" },
            { id: 2, timestamp: new Date(Date.now() - 3600000), 
              location: "Highway Exit 42", cameraId: "CAM-002", severity: "Medium", videoUrl: "sample2.mp4" }
        ];
    }

    function showView(view) {
        // Hide all content sections
        document.querySelectorAll('[id$="Content"]').forEach(section => {
            section.style.display = 'none';
        });

        // Show selected view
        document.getElementById(`${view}Content`).style.display = 'block';

        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.getElementById(`${view}Link`).classList.add('active');

        // Load view-specific data
        switch(view) {
            case 'dashboard':
                loadDashboard();
                break;
            case 'cameras':
                loadCameras();
                break;
            case 'accidents':
                loadAccidents();
                break;
            case 'settings':
                loadSettings();
                break;
        }
    }

    function loadDashboard() {
        // Update stats
        document.getElementById('activeCamerasCount').textContent = cameras.length;
        document.getElementById('todaysAccidentsCount').textContent = accidents.filter(a => {
            return new Date(a.timestamp).toDateString() === new Date().toDateString();
        }).length;
        document.getElementById('alertsSentCount').textContent = accidents.length * 2;

        // Populate accidents table
        const tableBody = document.getElementById('recentAccidentsTable');
        tableBody.innerHTML = accidents.length === 0 ? 
            '<tr><td colspan="5" class="text-center">No accidents detected yet</td></tr>' :
            accidents.slice(0, 5).map(accident => `
                <tr>
                    <td>${formatTime(accident.timestamp)}</td>
                    <td>${accident.location}</td>
                    <td>${accident.cameraId}</td>
                    <td><span class="badge bg-${getSeverityClass(accident.severity)}">${accident.severity}</span></td>
                    <td><button class="btn btn-sm btn-outline-primary view-accident" data-id="${accident.id}">View</button></td>
                </tr>
            `).join('');

        // Add event listeners to view buttons
        document.querySelectorAll('.view-accident').forEach(btn => {
            btn.addEventListener('click', function() {
                const accidentId = parseInt(this.getAttribute('data-id'));
                viewAccident(accidentId);
            });
        });
    }

    function loadCameras() {
        const cameraList = document.getElementById('cameraList');
        cameraList.innerHTML = cameras.length === 0 ? 
            '<div class="col-12"><div class="alert alert-info">No cameras configured yet</div></div>' :
            cameras.map(camera => `
                <div class="col-md-6 col-lg-4 mb-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5 class="card-title">${camera.name}</h5>
                            <p class="card-text"><i class="bi bi-geo-alt"></i> ${camera.location}</p>
                            <p class="card-text"><small class="text-muted">ID: ${camera.id}</small></p>
                        </div>
                        <div class="card-footer bg-transparent">
                            <button class="btn btn-sm btn-primary edit-camera" data-id="${camera.id}">Edit</button>
                            <button class="btn btn-sm btn-outline-danger delete-camera" data-id="${camera.id}">Delete</button>
                        </div>
                    </div>
                </div>
            `).join('');

        // Add event listeners to camera buttons
        document.querySelectorAll('.edit-camera').forEach(btn => {
            btn.addEventListener('click', function() {
                const cameraId = parseInt(this.getAttribute('data-id'));
                editCamera(cameraId);
            });
        });

        document.querySelectorAll('.delete-camera').forEach(btn => {
            btn.addEventListener('click', function() {
                const cameraId = parseInt(this.getAttribute('data-id'));
                deleteCamera(cameraId);
            });
        });
    }

    function loadAccidents() {
        const videosList = document.getElementById('accidentVideosList');
        videosList.innerHTML = accidents.length === 0 ? 
            '<div class="col"><div class="alert alert-info">No accident videos available</div></div>' :
            accidents.map(accident => `
                <div class="col">
                    <div class="card video-thumbnail">
                        <div class="card-body">
                            <h5 class="card-title">${accident.location}</h5>
                            <p class="card-text"><small class="text-muted">${formatTime(accident.timestamp)}</small></p>
                            <p class="card-text"><span class="badge bg-${getSeverityClass(accident.severity)}">${accident.severity}</span></p>
                        </div>
                        <div class="card-footer bg-transparent">
                            <button class="btn btn-sm btn-primary view-video" data-id="${accident.id}">
                                <i class="bi bi-play-fill me-1"></i> Play
                            </button>
                        </div>
                    </div>
                </div>
            `).join('');

        // Add event listeners to video buttons
        document.querySelectorAll('.view-video').forEach(btn => {
            btn.addEventListener('click', function() {
                const accidentId = parseInt(this.getAttribute('data-id'));
                viewAccident(accidentId);
            });
        });
    }

    function loadSettings() {
        // Load settings values
        document.getElementById('policeContact').value = '911';
        document.getElementById('hospitalContact').value = '112';
        document.getElementById('enableSMSAlerts').checked = true;
        document.getElementById('enableEmailAlerts').checked = true;
    }

    function editCamera(id) {
        const camera = cameras.find(c => c.id === id);
        if (!camera) return;

        document.getElementById('cameraId').value = camera.id;
        document.getElementById('cameraName').value = camera.name;
        document.getElementById('cameraLocation').value = camera.location;
        document.getElementById('cameraURL').value = camera.url;
        document.getElementById('cameraUsername').value = camera.username || '';
        document.getElementById('cameraPassword').value = camera.password || '';
        document.getElementById('cameraModalTitle').textContent = 'Edit Camera';
        cameraModal.show();
    }

    function saveCamera() {
        const id = parseInt(document.getElementById('cameraId').value) || 0;
        const name = document.getElementById('cameraName').value;
        const location = document.getElementById('cameraLocation').value;
        const url = document.getElementById('cameraURL').value;
        const username = document.getElementById('cameraUsername').value;
        const password = document.getElementById('cameraPassword').value;

        if (!name || !location || !url) {
            alert('Please fill in all required fields');
            return;
        }

        if (id) {
            // Update existing camera
            const index = cameras.findIndex(c => c.id === id);
            if (index !== -1) {
                cameras[index] = { id, name, location, url, username, password };
            }
        } else {
            // Add new camera
            const newId = cameras.length > 0 ? Math.max(...cameras.map(c => c.id)) + 1 : 1;
            cameras.push({ id: newId, name, location, url, username, password });
        }

        cameraModal.hide();
        loadCameras();
        loadDashboard();
    }

    function deleteCamera(id) {
        if (confirm('Are you sure you want to delete this camera?')) {
            cameras = cameras.filter(c => c.id !== id);
            loadCameras();
            loadDashboard();
        }
    }

    function viewAccident(id) {
        const accident = accidents.find(a => a.id === id);
        if (!accident) return;

        document.getElementById('videoModalTitle').textContent = `Accident at ${accident.location}`;
        document.getElementById('videoDetails').innerHTML = `
            <strong>Time:</strong> ${formatTime(accident.timestamp)}<br>
            <strong>Location:</strong> ${accident.location}<br>
            <strong>Camera ID:</strong> ${accident.cameraId}<br>
            <strong>Severity:</strong> <span class="badge bg-${getSeverityClass(accident.severity)}">${accident.severity}</span>
        `;

        const videoPlayer = document.getElementById('accidentVideoPlayer');
        videoPlayer.innerHTML = `<source src="${accident.videoUrl}" type="video/mp4">`;
        videoModal.show();
    }

    function downloadVideo() {
        alert('Video download would start here');
    }

    function searchVideos() {
        const query = document.getElementById('videoSearch').value.toLowerCase();
        const filtered = accidents.filter(accident => 
            accident.location.toLowerCase().includes(query) || 
            accident.cameraId.toLowerCase().includes(query)
        );
        loadAccidents(filtered);
    }

    function handleLogin(e) {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;

        if (email && password) {
            currentUser = {
                id: 1,
                name: "Admin",
                email: email
            };
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
            localStorage.setItem('authToken', 'simulated-token');
            loginModal.hide();
            showView('dashboard');
        } else {
            alert('Please enter email and password');
        }
    }

    function handleRegister(e) {
        e.preventDefault();
        const name = document.getElementById('registerName').value;
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        const confirmPassword = document.getElementById('registerConfirmPassword').value;

        if (password !== confirmPassword) {
            alert('Passwords do not match');
            return;
        }

        if (name && email && password) {
            currentUser = {
                id: 2,
                name: name,
                email: email
            };
            localStorage.setItem('currentUser', JSON.stringify(currentUser));
            localStorage.setItem('authToken', 'simulated-token');
            registerModal.hide();
            showView('dashboard');
        }
    }

    function saveSettings(e) {
        e.preventDefault();
        alert('Settings saved successfully');
    }

    function logout() {
        localStorage.removeItem('currentUser');
        localStorage.removeItem('authToken');
        currentUser = null;
        showLoginModal();
    }

    function showLoginModal() {
        loginModal.show();
    }

    // Helper functions
    function formatTime(timestamp) {
        return new Date(timestamp).toLocaleString();
    }

    function getSeverityClass(severity) {
        switch (severity.toLowerCase()) {
            case 'high': return 'danger';
            case 'medium': return 'warning';
            case 'low': return 'success';
            default: return 'secondary';
        }
    }
});