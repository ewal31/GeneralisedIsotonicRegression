{
  system ? builtins.currentSystem,
  nixpkgs ?
    fetchTarball "https://github.com/NixOS/nixpkgs/archive/057f9aecfb71c4437d2b27d3323df7f93c010b7e.tar.gz"
}:

let

pkgs = import nixpkgs {

  inherit system;
  config = {};
  overlays = [];
};

getFetchContentFlags = file:
  let
    inherit (builtins) head elemAt match;
    parse = match
      "(.*)(\n|^)FetchContent_Declare\\(\n *([^\n]*)\n([^)]*)\\).*"
      file;
    name = elemAt parse 2;
    content = elemAt parse 3;
    getKey = key: elemAt
      (match "(.*\n)? *${key} *\"?([^\n\"]*)(\"?\n.*)?" content) 1;
    url = getKey "GIT_REPOSITORY";
    pkg = if (head (match ".*(github|gitlab).com/([^/]*)/.*" url) == "github")
    then
      pkgs.fetchFromGitHub {
        inherit name;
        owner = elemAt (match ".*(github|gitlab).com/([^/]*)/.*" url) 1;
        repo = head (match ".*/([^/]*)($| |\\.git)" url);
        rev = getKey "GIT_TAG";
        hash = getKey "# hash:";
      }
    else
      pkgs.fetchFromGitLab {
        inherit name;
        owner = elemAt (match ".*(github|gitlab).com/([^/]*)/.*" url) 1;
        repo = head (match ".*/([^/]*)($| |\\.git)" url);
        rev = getKey "GIT_TAG";
        hash = getKey "# hash:";
      };
  in
    if (parse == null)
    then
      [ ]
    else
      ([ "-DFETCHCONTENT_SOURCE_DIR_${pkgs.lib.toUpper name}=${pkg}" ] ++
        getFetchContentFlags (head parse));

  buildType = "Release";

  commonCmakeFlags = getFetchContentFlags
      (builtins.readFile ./CMakeLists.txt);

  libCmakeFlags = getFetchContentFlags
      (builtins.readFile ./isotonic_regression/CMakeLists.txt);

  cliCmakeFlags = getFetchContentFlags
      (builtins.readFile ./cli/CMakeLists.txt);

  testsCmakeFlags = getFetchContentFlags
      (builtins.readFile ./tests/CMakeLists.txt);

  pythonCmakeFlags = getFetchContentFlags
      (builtins.readFile ./python/CMakeLists.txt);

  basename = "iso";
  version = builtins.head (
    builtins.match
    ".*VERSION +([0-9]+\\.[0-9]+\\.[0-9]+).*"
    (builtins.readFile ./CMakeLists.txt)
  );

in {

  # TODO should build lib separately and provide to others
  # gir = pkgs.stdenv.mkDerivation rec {
  #   inherit version;
  #   name = "${basename}-gir";
  #   pname = "${name}-${version}";
  #   src = builtins.path {
  #     inherit name;
  #     path = ./.;
  #   };
  #   buildInputs = with pkgs; [
  #     cmake
  #     llvmPackages_11.clang
  #   ];
  #   cmakeFlags = [
  #       "-DCMAKE_BUILD_TYPE=${buildType}"
  #       "-DBUILD_GIR_CLI_TOOL=ON"
  #     ] ++ commonCmakeFlags ++ libCmakeFlags ++ cliCmakeFlags;
  # };

  # tests = pkgs.stdenv.mkDerivation rec {
  #   inherit version;
  #   name = "${basename}-tests";
  #   pname = "${name}-${version}";
  #   src = builtins.path {
  #     inherit name;
  #     path = ./.;
  #   };
  #   buildInputs = with pkgs; [
  #     cmake
  #     llvmPackages_11.clang
  #   ];
  #   cmakeFlags = [
  #       "-DCMAKE_BUILD_TYPE=${buildType}"
  #       "-DBUILD_GIR_TESTS=ON"
  #     ] ++ commonCmakeFlags ++ libCmakeFlags ++ testsCmakeFlags;
  # };

  # example = pkgs.stdenv.mkDerivation rec {
  #   inherit version;
  #   name = "${basename}-example";
  #   pname = "${name}-${version}";
  #   src = builtins.path {
  #     inherit name;
  #     path = ./.;
  #   };
  #   buildInputs = with pkgs; [
  #     cmake
  #     llvmPackages_11.clang
  #   ];
  #   cmakeFlags = [
  #       "-DCMAKE_BUILD_TYPE=${buildType}"
  #       "-DBUILD_GIR_EXAMPLES=ON"
  #     ] ++ commonCmakeFlags ++ libCmakeFlags;
  # };

  python = pkgs.stdenv.mkDerivation rec {
    inherit version;
    name = "${basename}-python";
    pname = "${name}-${version}";
    format = "setuptools"; # TODO pyproject doesn't include directories properly

    src = builtins.path {
      inherit name;
      path = ./.;
    };

    nativeBuildInputs = with pkgs; [
      cmake
      llvmPackages_11.clang
    ];

    buildInputs = with pkgs; [
    ];

    propagatedBuildInputs = with pkgs; [
      python311
      python311.pkgs.numpy
      python311.pkgs.setuptools
    ];

    cmakeFlags = [
        "-DCMAKE_BUILD_TYPE=${buildType}"
        "-DBUILD_GIR_PYTHON=ON"
      ] ++ commonCmakeFlags ++ libCmakeFlags ++ pythonCmakeFlags;

    preBuild = ''
      cd ..
    '';

  };

  # wasm = pkgs.stdenv.mkDerivation rec {
  #   inherit version;
  #   name = "${basename}-example";
  #   pname = "${name}-${version}";
  #   src = builtins.path {
  #     inherit name;
  #     path = ./.;
  #   };
  #   buildInputs = with pkgs; [
  #     cmake
  #     llvmPackages_11.clang
  #     emscripten
  #   ];

  #   cmakeFlags = [
  #       "-DCMAKE_BUILD_TYPE=${buildType}"
  #       "-DBUILD_GIR_EXAMPLES=ON"
  #     ] ++ commonCmakeFlags ++ libCmakeFlags;
  # };

}
