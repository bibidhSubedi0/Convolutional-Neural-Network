workspace "ConvolutionalNeuralNetwork"
    configurations { "Debug", "Release" }
    architecture "x64"

project "ConvolutionalNeuralNetwork"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++20"  -- Upgraded to C++20
    targetdir "bin/%{cfg.buildcfg}"
    objdir "bin-int/%{cfg.buildcfg}"

    files { "src/**.cpp", "include/cnn/**.hpp", "include/cnn/**.h" }
    includedirs { "include/cnn" }

    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
