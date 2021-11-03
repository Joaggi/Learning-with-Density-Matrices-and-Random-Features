

def initial_configuration:
    import os 
    try:  
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        import os
        os.system("pip3 install mlflow")

        from google.colab import drive
        drive.mount('/content/drive')
        os.chdir('/content/drive/MyDrive/Academico/doctorado_programacion/experiments/2021_01_learning_with_density_matrices')
        import sys
        sys.path.append('submodules/qmc/')
        #sys.path.append('../../../../submodules/qmc/')
        print(sys.path)
    else:
        import sys
        sys.path.append('submodules/qmc/')
        sys.path.append('data/')
        #sys.path.append('../../../../submodules/qmc/')
        print(sys.path)
        # %cd ../../

    print(os.getcwd())


