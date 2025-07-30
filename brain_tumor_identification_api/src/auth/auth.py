# app/auth/auth.py (corrected)
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required

# Create the Blueprint for authentication routes
auth_bp = Blueprint('auth', __name__, url_prefix='/login')

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'
login_manager.login_message = "Please log in to access this page."

users = {}
_bcrypt = None

class User(UserMixin):
    def __init__(self, id, username, password_hash, roles):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.roles = roles

    def has_role(self, role):
        return role in self.roles

# This function is DEFINED here, not imported from itself.
def init_auth(app, bcrypt):
    login_manager.init_app(app)
    global users, _bcrypt
    _bcrypt = bcrypt
    users.update({
        "1": {"username": "admin", "password": bcrypt.generate_password_hash("admin").decode('utf-8'), "roles": ["admin"]},
        "2": {"username": "doctor", "password": bcrypt.generate_password_hash("doctor").decode('utf-8'), "roles": ["doctor"]},
        "3": {"username": "radiologist", "password": bcrypt.generate_password_hash("radiologist").decode('utf-8'), "roles": ["radiologist", "doctor"]},
    })
    print("Users initialized:", {uid: data["username"] for uid, data in users.items()})

@login_manager.user_loader
def load_user(user_id):
    user_data = users.get(user_id)
    if user_data:
        return User(
            id=user_id,
            username=user_data["username"],
            password_hash=user_data["password"],
            roles=user_data["roles"]
        )
    return None

# --- Authentication Routes ---

@auth_bp.route('/', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user by username more efficiently
        user_data_tuple = next(((uid, data) for uid, data in users.items() if data["username"] == username), None)

        user_to_check = None
        if user_data_tuple:
            uid, user_data = user_data_tuple
            user_to_check = User(uid, user_data["username"], user_data["password"], user_data["roles"])

        # Use the bcrypt instance to check the password hash for consistency
        if user_to_check and _bcrypt.check_password_hash(user_to_check.password_hash, password):
            login_user(user_to_check)
            flash(f'Logged in as {user_to_check.username}.', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.index'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))