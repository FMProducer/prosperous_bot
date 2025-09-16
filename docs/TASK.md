*** Begin Patch
*** Update File: .github/workflows/ci.yml
@@
   steps:
     - name: Checkout
       uses: actions/checkout@v4
+
+    # Create pip cache directory early so actions/setup-python can save cache on post-run
+    - name: Prepare pip cache dir
+      shell: bash
+      run: mkdir -p "$PIP_CACHE_DIR"
 
     - name: Setup Python ${{ matrix.python-version }}
       uses: actions/setup-python@v5
       with:
         python-version: ${{ matrix.python-version }}
         cache: 'pip'
         cache-dependency-path: requirements.txt
*** End Patch
