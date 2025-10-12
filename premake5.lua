workspace "ConvolutionalNeuralNetwork"
    configurations { "Debug", "Release" }
    architecture "x64"

project "ConvolutionalNeuralNetwork"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++20"
    targetdir "bin/%{cfg.buildcfg}"
    objdir "bin-int/%{cfg.buildcfg}"

    files { "srcs/**.cpp", "includes/cnn/**.hpp", "includes/cnn/**.h" }
    includedirs {
        "includes/cnn",
        "D:/opencv/build/include"       
    }

    libdirs { "D:/opencv/build/x64/vc16/lib" }

    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
        links {
            "opencv_world4100d.lib"
        }

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"
        links {
            "opencv_world4100.lib"
        }
