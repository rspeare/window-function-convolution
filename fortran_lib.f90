  subroutine shellSumFortran(phi1,phi2,phi3,s)
    implicit none
    real, intent(in):: phi1(:), phi2(:), phi3(:)
    real*8,intent(out):: s
    integer :: i
    s=0
    !$OMP PARALLEL DO private(i) shared(phi1,phi2,phi3,sum)
!    s=SUM(phi1(:)*phi2(:)*phi3(:))
    do i=size(phi1),1,-1
       s=s+phi1(i)*phi2(i)*phi3(i)
    enddo
  end subroutine shellSumFortran



