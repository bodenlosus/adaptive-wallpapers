{
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      with nixpkgs.legacyPackages.${system};
      let
        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);

        pkg = python3.pkgs.buildPythonPackage rec {
          pname = pyproject.project.name;
          version = pyproject.project.version;
          format = "pyproject";
          src = ./.;

          nativeBuildInputs = with python3.pkgs; [ setuptools ];

          propagatedBuildInputs = with python3.pkgs; [
            numba
            pillow
            numpy
            scikit-learn
          ];

          installPhase = ''
            runHook preInstall
            # Add debugging to inspect the build environment
            runHook postInstall
          '';

          installCheckInputs = with python3.pkgs; [
          ];
        };

        editablePkg = pkg.overrideAttrs (oldAttrs: {
          nativeBuildInputs = oldAttrs.nativeBuildInputs ++ [
            (python3.pkgs.mkPythonEditablePackage {
              pname = pyproject.project.name;
              inherit (pyproject.project) scripts version;
              root = "$PWD";
            })
          ];
        });

      in {
        packages.default = pkg;
        devShells.default = mkShell {
          venvDir = "./.venv";
          packages = with python3.pkgs; [
          numba
          pillow
          numpy
          scikit-learn
          venvShellHook
          ];
          inputsFrom = [ editablePkg ];
        };
      });
}