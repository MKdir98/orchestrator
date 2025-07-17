from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from orchestrator.models import User
from orchestrator.services.user_service import UserService

class UserSettingsTest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = UserService.create_user("test_user", "test@example.com", "password123")
        self.client.force_authenticate(user=self.user)
        
    def test_update_continue_automatically(self):
        """تست به‌روزرسانی تنظیم continue_automatically"""
        url = reverse('user-settings', args=[self.user.id])
        data = {
            "continue_automatically": True
        }
        
        response = self.client.patch(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['continue_automatically'])
        
        # تست تغییر مجدد به false
        data = {
            "continue_automatically": False
        }
        response = self.client.patch(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data['continue_automatically'])
        
    def test_invalid_user_id(self):
        """تست با شناسه کاربر نامعتبر"""
        url = reverse('user-settings', args=[999])
        data = {
            "continue_automatically": True
        }
        
        response = self.client.patch(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        
    def test_unauthorized_access(self):
        """تست دسترسی بدون احراز هویت"""
        self.client.force_authenticate(user=None)
        url = reverse('user-settings', args=[self.user.id])
        data = {
            "continue_automatically": True
        }
        
        response = self.client.patch(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        
    def test_invalid_data(self):
        """تست با داده نامعتبر"""
        url = reverse('user-settings', args=[self.user.id])
        data = {
            "continue_automatically": "invalid"
        }
        
        response = self.client.patch(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST) 