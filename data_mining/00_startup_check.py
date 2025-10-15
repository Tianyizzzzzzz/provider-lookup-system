# ============================================
# 文件 1: data_mining/00_startup_check.py
# 项目启动验证脚本
# ============================================
"""
Project Startup Verification Script
验证所有项目组件是否正常运行
"""

import os
import sys
import subprocess


def check_environment():
    """检查基本环境"""
    print("=" * 60)
    print("🔍 PROJECT STARTUP VERIFICATION")
    print("=" * 60)

    # 1. 检查Python版本
    print("\n1. Python Environment:")
    python_version = sys.version
    print(f"   Python Version: {python_version}")

    # 2. 检查虚拟环境
    in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    print(f"   Virtual Environment: {'✅ Active' if in_venv else '❌ Not Active'}")

    if not in_venv:
        print("   ⚠️  Please activate virtual environment: .venv\\Scripts\\activate")
        return False

    # 3. 检查必要的包
    print("\n2. Required Packages:")
    required_packages = [
        ('django', 'django'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('psycopg2', 'psycopg2'),
        ('python-decouple', 'decouple')  # Package name vs import name
    ]

    all_installed = True
    for display_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} - NOT INSTALLED")
            all_installed = False

    if not all_installed:
        print("\n   Install missing packages:")
        print("   pip install -r provider_lookup_web/requirements.txt")
        return False

    # 4. 检查Django项目路径
    print("\n3. Project Structure:")
    project_root = os.getcwd()
    django_path = os.path.join(project_root, 'provider_lookup_web')

    print(f"   Current Directory: {project_root}")
    print(f"   Django App Path: {django_path}")
    print(f"   Django App Exists: {'✅ Yes' if os.path.exists(django_path) else '❌ No'}")

    # 5. 检查环境变量文件
    env_file = os.path.join(django_path, '.env')
    print(f"   .env File: {'✅ Found' if os.path.exists(env_file) else '⚠️  Not Found'}")

    return True


def check_database_connection():
    """检查数据库连接"""
    print("\n4. Database Connection:")

    try:
        # Setup Django
        sys.path.insert(0, 'provider_lookup_web')
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')

        import django
        django.setup()

        from django.db import connection

        # Test connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

        print("   ✅ Database connection successful")

        # Check database name
        db_name = connection.settings_dict['NAME']
        print(f"   Database: {db_name}")

        return True

    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        print("\n   Troubleshooting:")
        print("   1. Check if PostgreSQL is running: net start postgresql-x64-15")
        print("   2. Verify .env file database credentials")
        print("   3. Test connection: psql -U postgres -d provider_lookup")
        return False


def check_django_models():
    """检查Django模型"""
    print("\n5. Django Models:")

    try:
        from apps.providers.models import Provider, ProviderAddress, ProviderTaxonomy

        provider_count = Provider.objects.count()
        address_count = ProviderAddress.objects.count()
        taxonomy_count = ProviderTaxonomy.objects.count()

        print(f"   Providers: {provider_count:,} ✅")
        print(f"   Addresses: {address_count:,} ✅")
        print(f"   Taxonomies: {taxonomy_count:,} ✅")

        if provider_count == 0:
            print("   ⚠️  No provider data found - database may need to be populated")
            return False

        return True

    except Exception as e:
        print(f"   ❌ Model check failed: {e}")
        return False


def main():
    """主函数"""
    checks = [
        ("Environment Setup", check_environment),
        ("Database Connection", check_database_connection),
        ("Django Models", check_django_models)
    ]

    all_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
            break

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Project is ready!")
        print("\nNext steps:")
        print("  1. Run diagnostic: python data_mining\\diagnostic_check.py")
        print("  2. Run data exploration: python data_mining\\01_data_exploration.py")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
    print("=" * 60)


if __name__ == "__main__":
    main()