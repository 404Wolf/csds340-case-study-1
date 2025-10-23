{
  description = "CSDS340 Case Study 1";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3.withPackages (
          ps: with ps; [
            black
            scikit-learn
            matplotlib
            numpy
            pandas
            sentence-transformers
          ]
        );

        treefmtconfig = inputs.treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs = {
            black.enable = true;
            prettier.enable = true;
          };
        };
      in
      {
        devShells = {
          TYPST_FONT_PATHS = "${pkgs.liberation-sans-narrow}/share/fonts/truetype";
          default = pkgs.mkShell {
            packages = [
              pkgs.typst
              pkgs.pyright
              pkgs.nil
              pkgs.nixd
              pkgs.mermaid-cli
              python
            ];
          };
        };
        formatter = treefmtconfig.config.build.wrapper;
        checks = {
          formatting = treefmtconfig.build.check self;
        };
      }
    );
}
