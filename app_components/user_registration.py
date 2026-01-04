"""
Simple one-time user registration component for GeoClimate Fetcher
Tracks basic user information without being intrusive
"""
import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import hashlib


class UserRegistration:
    """Handle one-time user registration and tracking"""

    def __init__(self, storage_file="user_data.json"):
        """Initialize user registration tracker"""
        self.storage_file = Path(storage_file)

    def get_user_hash(self, project_id):
        """Create anonymous user hash from project ID"""
        return hashlib.sha256(project_id.encode()).hexdigest()[:16]

    def is_registered(self, project_id):
        """Check if user has already registered"""
        user_hash = self.get_user_hash(project_id)

        # Check session state first
        if st.session_state.get('user_registered', False):
            return True

        # Check stored data
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    return user_hash in data.get('users', {})
            except:
                return False
        return False

    def save_user_data(self, project_id, user_info):
        """Save user information to JSON file"""
        user_hash = self.get_user_hash(project_id)

        # Load existing data
        data = {'users': {}, 'stats': {'total_users': 0}}
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
            except:
                pass

        # Add user data with timestamp
        data['users'][user_hash] = {
            **user_info,
            'registered_at': datetime.now().isoformat(),
            'user_id': user_hash
        }

        # Update stats
        data['stats']['total_users'] = len(data['users'])

        # Save to file
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save user data: {e}")

    def get_stats(self):
        """Get user statistics"""
        if not self.storage_file.exists():
            return {'total_users': 0, 'countries': {}, 'purposes': {}}

        try:
            with open(self.storage_file, 'r') as f:
                data = json.load(f)
                users = data.get('users', {})

                stats = {
                    'total_users': len(users),
                    'countries': {},
                    'purposes': {},
                    'institutions': {}
                }

                # Aggregate stats
                for user_data in users.values():
                    country = user_data.get('country', 'Unknown')
                    purpose = user_data.get('purpose', 'Unknown')
                    institution = user_data.get('institution', 'Unknown')

                    stats['countries'][country] = stats['countries'].get(country, 0) + 1
                    stats['purposes'][purpose] = stats['purposes'].get(purpose, 0) + 1
                    if institution != 'Unknown' and institution:
                        stats['institutions'][institution] = stats['institutions'].get(institution, 0) + 1

                return stats
        except:
            return {'total_users': 0, 'countries': {}, 'purposes': {}}

    def render(self, project_id):
        """
        Render registration form
        Returns True if user is registered (already or just completed)
        """
        # Check if already registered
        if self.is_registered(project_id):
            st.session_state.user_registered = True
            return True

        # Show registration form
        st.markdown("""
        <style>
        .registration-card {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 20px;
            padding: 2.5rem;
            margin: 2rem 0;
            box-shadow: 0 10px 40px rgba(8, 131, 149, 0.2);
            border: 1px solid rgba(8, 131, 149, 0.3);
        }

        @media (prefers-color-scheme: dark) {
            .registration-card {
                background: linear-gradient(135deg, #1a2332 0%, #243447 100%);
                border: 1px solid rgba(8, 131, 149, 0.4);
            }
        }

        .reg-title {
            font-size: 2rem;
            font-weight: 600;
            color: #088395;
            text-align: center;
            margin-bottom: 1rem;
        }

        .reg-subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
            font-size: 1rem;
        }

        @media (prefers-color-scheme: dark) {
            .reg-subtitle {
                color: #ccc;
            }
        }

        .privacy-note {
            background: rgba(8, 131, 149, 0.1);
            border-left: 4px solid #088395;
            padding: 1rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            font-size: 0.9rem;
        }
        </style>

        <div class="registration-card">
            <h2 class="reg-title">👋 Welcome to GeoClimate!</h2>
            <p class="reg-subtitle">
                Help us understand our community better (Optional - Takes 30 seconds)
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("user_registration"):
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input(
                    "Your Name (Optional)",
                    placeholder="e.g., Dr. Jane Smith",
                    help="We'd love to know who's using our platform!"
                )

                institution = st.text_input(
                    "Institution/Organization (Optional)",
                    placeholder="e.g., University, Research Center, Company",
                    help="Where are you from?"
                )

            with col2:
                country = st.selectbox(
                    "Country (Optional)",
                    [""] + [
                        "United States", "United Kingdom", "Canada", "Australia", "India",
                        "Germany", "France", "Netherlands", "Brazil", "China", "Japan",
                        "South Korea", "Singapore", "Nepal", "Bangladesh", "Pakistan",
                        "Other"
                    ],
                    help="Help us understand our global reach"
                )

                purpose = st.selectbox(
                    "Primary Use Case (Optional)",
                    [""] + [
                        "Academic Research",
                        "Student Project",
                        "Climate Analysis",
                        "Hydrology Study",
                        "Environmental Monitoring",
                        "Policy Making",
                        "NGO/Non-profit Work",
                        "Personal Learning",
                        "Other"
                    ],
                    help="How will you use GeoClimate?"
                )

            st.markdown("""
            <div class="privacy-note">
                🔒 <strong>Privacy First:</strong> All fields are optional. We store minimal,
                anonymized data to understand our user base. No personal information is shared
                with third parties. You can skip this entirely and use the platform freely.
            </div>
            """, unsafe_allow_html=True)

            col_submit, col_skip = st.columns([1, 1])

            with col_submit:
                submitted = st.form_submit_button(
                    "✅ Submit & Continue",
                    type="primary",
                    use_container_width=True
                )

            with col_skip:
                skipped = st.form_submit_button(
                    "⏭️ Skip for Now",
                    use_container_width=True
                )

        if submitted or skipped:
            # Save user data (even if empty fields - counts as registered)
            user_info = {
                'name': name if name else 'Anonymous',
                'institution': institution if institution else None,
                'country': country if country else 'Not specified',
                'purpose': purpose if purpose else 'Not specified',
                'skipped': skipped
            }

            self.save_user_data(project_id, user_info)
            st.session_state.user_registered = True

            if submitted:
                st.success("🎉 Thank you! Your response helps us improve GeoClimate.")
            else:
                st.info("👍 No problem! You can always provide feedback later.")

            st.rerun()

        return False
