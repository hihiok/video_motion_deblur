"""Authoritative v3 migration entry point; old runs are never overwritten."""
from .prepare_crop import migrate, main

if __name__ == '__main__':
    main()
