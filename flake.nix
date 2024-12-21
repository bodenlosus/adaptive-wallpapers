{
  description = "DevShell and package for Python project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          venvDir = ".venv";
          packages = with pkgs; [ python312 ] ++
            (with pkgs.python312Packages; [
              pip
              numba
              pillow
              numpy
              scikit-learn
              venvShellHook
            ]);
        };
      });

      packages = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.python312Packages.buildPythonApplication rec {
          pname = "adaptive-wallpapers";
          version = "0.1.0";

          # Specify the source directory
          propagatedBuildInputs = with pkgs.python3Packages; [
            numba
            pillow
            numpy
          ];

          src = ./.;

          meta = with pkgs.lib; {
            description = "A Python application with CLI functionality";
            license = licenses.mit;
            maintainers = with maintainers; [ your_name_here ];
          };
        };
      });
    };
}
